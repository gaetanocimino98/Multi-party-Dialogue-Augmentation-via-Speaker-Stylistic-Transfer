# The LLMGenerator class implements the core EDU generation and refinement pipeline 
# for MIMIC-style multi-party dialogue augmentation. Specifically, it handles the 
# following responsibilities:
#
# 1. Constructs discourse-aware prompts by incorporating rhetorical relation 
#    explanations, dialogue context, and stylistic samples from target and 
#    interacting speakers. This supports generation of structurally consistent 
#    rephrasings using LLMs.
#
# 2. Utilizes an LLM to rephrase selected EDUs from a target speaker (speaker A) 
#    in the style of a substitute speaker (speaker B), preserving discourse coherence.
#
# 3. Integrates interpersonal cues by extracting past interaction patterns 
#    from speakers involved in discourse relations with the target EDU.
#    These are used to inform the LLM of stylistic and contextual expectations.
#
# 4. Supports discourse relation validation using BERT-based binary classifiers
#    trained per relation type. These classifiers assess whether generated EDUs
#    still satisfy the rhetorical constraints.
#
# 5. If any relation fails validation, the system collects suggestions and invokes
#    an external refinement agent to iteratively adjust the EDU while 
#    preserving intended relations. This realizes the EDU Refinement stage of MIMIC.
#
# This implementation embodies the full MIMIC pipeline, combining MASK-driven
# speaker selection, MIRROR-based stylistic substitution, 
# interpersonal modeling, and discourse-constrained generation and validation.

from openai import OpenAI
from refinement import Agent
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification


class LLMGenerator:

    def __init__(self, relation_explanations, edu_to_replace, constraints,
                 b_relations, interpersonal,
                 model=None, base_url=None, api_key=None, client=None):

        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-instruct")

        self.client = client or OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY", "not-needed"),
        )

        self.relation_explanations = relation_explanations
        self.edu_to_replace = edu_to_replace
        self.constraints = constraints
        self.b_relations = b_relations
        self.interpersonal = interpersonal
        self.refiner = Agent(self.model, relation_explanations, client=self.client)

        self._bert_cache = {}
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __predict(self, relation, sentence1, sentence2, max_len=128):

        if relation not in self._bert_cache:
            path = f'bert_model_{relation}'
            model = BertForSequenceClassification.from_pretrained(path).to(self._device)
            model.eval()
            tokenizer = BertTokenizer.from_pretrained(path)
            self._bert_cache[relation] = (model, tokenizer)

        model, tokenizer = self._bert_cache[relation]

        encoding = tokenizer(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self._device)
        attention_mask = encoding['attention_mask'].to(self._device)
        token_type_ids = encoding['token_type_ids'].to(self._device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        return prediction

    def __system_prompt_definition(self, relations, relation_explanations):

        relation_explanations_filtered = []

        for related_utterance, relation_type, position, interaction_speaker_name in relations:
            relation_explanations_filtered.append(relation_explanations[relation_type])

        relations_string = '\n'.join(relation_explanations_filtered)

        system_prompt = f"""
        You are an advanced assistant specializing in transforming dialogues while preserving the coherence of discourse relations. 
        You will be provided with a structured dialogue where speaker A's utterances interact with one or more other speakers, creating multiple discourse relations.

        Your task is to rephrase a specific utterance from speaker A according to the linguistic style of speaker B while also considering the speaking tendencies of the other participants in the dialogue.

        Your primary objectives are:
        1. Rephrase speaker A's utterance using speaker B's typical language patterns, ensuring that speaker B's linguistic style is preserved.
        2. The utterance should be rewritten based on examples of speaker B's past interactions.
        3. Maintain all discourse relations involving the rephrased utterance. If speaker A's utterance is part of multiple relations, ensure the rewritten version satisfies all of them.
        4. If linguistic tendencies of the other speakers are provided, take them into account to ensure the rephrased utterance fits naturally within the broader conversational dynamics. Otherwise, focus solely on speaker B's past interactions.
        5. Ensure that the logical coherence and meaning of the conversation are preserved.

        You will be provided with:
        - The original dialogue, including all relevant interactions, where one specific utterance by speaker A needs to be rephrased.
        - Examples of speaker B's past interactions categorized by discourse relation type.
        - (Optional) Examples of how the other speakers in the dialogue typically communicate, to account for their linguistic tendencies.

        For each relation type, a brief explanation is provided to help you preserve its meaning:

        {relations_string}    
        """

        return system_prompt

    def __format_relation_sentence_A(self, utterance1, utterance2, relation, loc, interaction_speaker_name):
        article = "an" if relation[0].lower() in "aeiou" else "a"

        if loc == 'right_side':
            return f'The utterance "{utterance1}" by speaker A is in {article} "{relation}" relation with the utterance "{utterance2}" by speaker {interaction_speaker_name}.'
        elif loc == 'left_side':
            return f'The utterance "{utterance2}" by speaker {interaction_speaker_name} is in {article} "{relation}" relation with the utterance "{utterance1}" by speaker A.'
        elif loc == 'both-right':
            return f'The utterance "{utterance1}" by speaker A is in {article} "{relation}" relation with the utterance "{utterance2}" by speaker A.'
        else:
            return f'The utterance "{utterance2}" by speaker A is in {article} "{relation}" relation with the utterance "{utterance1}" by speaker A.'

    def __format_relation_sentence_B(self, utterance1, utterance2, relation, loc):
        article = "an" if relation[0].lower() in "aeiou" else "a"

        if loc == 'left_side':
            return f'The utterance "{utterance1}" by speaker B is in {article} "{relation}" relation with the utterance "{utterance2}".'
        elif loc == 'right_side':
            return f'The utterance "{utterance2}" is in {article} "{relation}" relation with the utterance "{utterance1}" by speaker B.'
        else:
            return f'The utterance "{utterance1}" by speaker B is in {article} "{relation}" relation with the utterance "{utterance2}" by speaker B.'

    def __user_prompt_definition(self, edu, a_relations, b_relations):

        speaker_a_samples = []
        speaker_b_samples = []
        speaker_a_samples_for_validation = []
        interpersonal_speakers_samples = {}

        considered_relations = []

        for related_utterance, relation_type, position, interaction_speaker_name in a_relations:
            speaker_a_samples.append(self.__format_relation_sentence_A(edu, related_utterance, relation_type,
                                                                       position, interaction_speaker_name))
            if interaction_speaker_name:
                interpersonal_speakers_samples.setdefault(interaction_speaker_name, []).append([relation_type, position])

            if position == 'left_side':
                speaker_a_samples_for_validation.append([related_utterance, "xxx", relation_type])
                key = (relation_type, 'right_side')
                if key not in considered_relations:
                    values = b_relations.get(key, [])
                    for entry in values:
                        speaker_b_samples.append(self.__format_relation_sentence_B(entry[0]['edu2'], entry[0]['edu1'],
                                                                                   entry[0]['rel_type'], entry[0]['role']))
                    considered_relations.append(key)
            elif position == 'right_side':
                speaker_a_samples_for_validation.append(["xxx", related_utterance, relation_type])
                key = (relation_type, 'left_side')
                if key not in considered_relations:
                    values = b_relations.get(key, [])
                    for entry in values:
                        speaker_b_samples.append(self.__format_relation_sentence_B(entry[0]['edu1'], entry[0]['edu2'],
                                                              entry[0]['rel_type'], entry[0]['role']))
                    considered_relations.append(key)
            else:
                if position == 'both-right':
                    speaker_a_samples_for_validation.append(["xxx", related_utterance, relation_type])
                else:
                    speaker_a_samples_for_validation.append([related_utterance, "xxx", relation_type])

                key = (relation_type, 'both')
                if key not in considered_relations:
                    values = b_relations.get(key, [])
                    for entry in values:
                        speaker_b_samples.append(self.__format_relation_sentence_B(entry[0]['edu1'], entry[0]['edu2'],
                                                                                   entry[0]['rel_type'], entry[0]['role']))
                    considered_relations.append(key)

        speaker_a_utterances = '\n'.join(speaker_a_samples)
        speaker_b_utterances = '\n'.join(speaker_b_samples)

        interpersonal_sentences = self.interpersonal.create_interpersonal_dictionary(interpersonal_speakers_samples)

        user_prompt = f"""
        Here is a structured dialogue where speaker A's utterance needs to be rephrased according to the linguistic style of speaker B. Ensure that speaker B's rewritten utterance matches the provided examples of their past interactions and preserves all discourse relations in the original dialogue.

        If examples of the linguistic tendencies of the other speakers are provided, ensure the rephrased utterance also aligns with these interactional nuances. If not, focus solely on reproducing speaker B's style.

        Original Dialogue:

        {speaker_a_utterances}

        Examples of speaker B's Style:

        {speaker_b_utterances}

        Examples of the linguistic tendencies of the other speakers:

        {interpersonal_sentences}

        Please rephrase the utterance "{edu}" by speaker A while ensuring that:
        - Speaker B's linguistic style is maintained.
        - All discourse relations in the original dialogue remain intact.
        - The relational and interpersonal dynamics in the dialogue are preserved when possible.

        Generate only the rephrased utterance, without any explanation or reasoning.
        """

        return user_prompt, speaker_b_utterances, interpersonal_sentences, speaker_a_samples_for_validation

    def edu_generation(self):

        generated_edus_refined = []

        for edu_id, edu in self.edu_to_replace:
            a_relations = self.constraints[edu_id]

            system_prompt = self.__system_prompt_definition(a_relations, self.relation_explanations)
            user_prompt, _, _, speaker_a_samples_for_validation = self.__user_prompt_definition(
                edu, a_relations, self.b_relations)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
            )

            generated_edu = completion.choices[0].message.content.strip('"')

            for i in range(2):
                suggestions = []
                flag = True

                for entry in speaker_a_samples_for_validation:
                    if entry[0] == "xxx":
                        prediction = self.__predict(entry[2], generated_edu, entry[1])
                        if prediction == 0:
                            suggestions.append(self.refiner.evaluation(edu, entry[1], entry[2], generated_edu))
                            flag = False
                    else:
                        prediction = self.__predict(entry[2], entry[0], generated_edu)
                        if prediction == 0:
                            suggestions.append(self.refiner.evaluation(edu, entry[0], entry[2], generated_edu))
                            flag = False

                if flag:
                    break
                else:
                    generated_edu = self.refiner.refine_EDU(system_prompt, user_prompt, suggestions)

            generated_edus_refined.append(generated_edu)

        return generated_edus_refined

