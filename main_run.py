from selection import SpeakerSelector
from replacement import SpeakerChanger
from llm_generation import LLMGenerator
import random, support, json, copy
from interpersonal import InterpersonalSelector
from tqdm import tqdm

corpus_path = "stac/molweni"

data = support.open_corpus(corpus_path)

synthetic_corpus_refined = []

for i in tqdm(range(len(data))):

    try:
        dialogue_id = i

        standard_number_in_context_examples = 5

        speaker_number = 0

        selector = SpeakerSelector(data, dialogue_id)

        final_scores = selector.compute_speaker_score()

        final_scores.sort(key=lambda x: x['score'], reverse=True)

        speaker_to_be_replaced = final_scores[speaker_number]['Name']

        speaker_names = [entry['Name'] for entry in final_scores]

        changer = SpeakerChanger(data, dialogue_id, speaker_to_be_replaced, speaker_names, standard_number_in_context_examples)

        replacement_speaker_name, replacement_speaker_relations, relation_types, number_of_examples = changer.select_random_name()

        edu_to_replace, constraints = changer.edu_constraints()

        considered_relations = []

        selected_examples = {}
        for keyword, lists in replacement_speaker_relations.items():
            if keyword not in considered_relations:
                if number_of_examples > 0:
                    if len(lists) > number_of_examples:
                        selected_examples[keyword] = random.sample(lists, number_of_examples)
                    else:
                        selected_examples[keyword] = lists
                else:
                    if len(lists) > standard_number_in_context_examples:
                        selected_examples[keyword] = random.sample(lists, standard_number_in_context_examples)
                    else:
                        selected_examples[keyword] = lists

                considered_relations.append(keyword)

        stac_relations = {
            "Comment": "A Comment relation typically indicates that one utterance provides a comment or opinion on the content of another utterance. It shows a speaker's perspective or evaluation of the preceding statement.",
            "Clarification_question": "In a Clarification-Question relation, one utterance poses a question seeking clarification or additional information about the content of another utterance. It implies a request for further explanation.",
            "Elaboration": "An Elaboration relation signifies that one utterance expands upon or provides more details about the content of another utterance. It is used to enhance understanding by offering additional information or context.",
            "Acknowledgement": "An Acknowledgment relation indicates that one utterance acknowledges or recognizes the content of another utterance. It signifies that the speaker has taken note of what was said.",
            "Continuation": "A Continuation relation suggests that one utterance continues the topic or discussion from a previous utterance. It signifies a logical progression in the conversation.",
            "Explanation": "An Explanation relation pertains to one utterance offering an explanation or clarification in response to a question or confusion expressed in another utterance. It aids in providing clarity.",
            "Conditional": "A Conditional relation implies that one utterance presents a condition or hypothetical scenario related to the content of another utterance. It often involves 'if-then' statements.",
            "Question_answer_pair": "A Question-Answer Pair relation indicates that one utterance contains a question, and another utterance immediately follows with an answer to that question. It demonstrates a direct question-and-answer interaction.",
            "Alternation": "An Alternation relation shows that two utterances present alternative options or choices. It is used when discussing multiple possibilities or courses of action.",
            "Q_Elab": "A Q-Elab relation signifies that one utterance asks a question, and another utterance follows with an elaboration or further explanation of the question or its context.",
            "Result": "A Result relation indicates that one utterance discusses the outcome or consequence of the content presented in another utterance. It shows a cause-and-effect relation.",
            "Background": "In a Background relation, one utterance provides background information or context that is relevant to the content of another utterance. It helps set the stage for the discussion.",
            "Narration": "A Narration relation holds when the main eventualities of two utterances occur in sequence.",
            "Correction": "A Correction relation shows that one utterance corrects or revises the content of another utterance. It is used to rectify errors or inaccuracies.",
            "Parallel": "A Parallel relation relations occur when two or more utterances share similar or related content, often in a parallel or analogous manner. It emphasizes similarities or comparisons.",
            "Contrast": "A Contrast relation signifies that one utterance presents content that is in contrast or opposition to the content of another utterance. It highlights differences or contradictions in the conversation."
        }

        """
        molweni_relations = {
            "Comment": "A Comment relation typically indicates that one utterance provides a comment or opinion on the content of another utterance. It shows a speaker's perspective or evaluation of the preceding statement.",
            "Clarification_question": "In a Clarification-Question relation, one utterance poses a question seeking clarification or additional information about the content of another utterance. It implies a request for further explanation.",
            "Elaboration": "An Elaboration relation signifies that one utterance expands upon or provides more details about the content of another utterance. It is used to enhance understanding by offering additional information or context.",
            "Acknowledgement": "An Acknowledgment relation indicates that one utterance acknowledges or recognizes the content of another utterance. It signifies that the speaker has taken note of what was said.",
            "Continuation": "A Continuation relation suggests that one utterance continues the topic or discussion from a previous utterance. It signifies a logical progression in the conversation.",
            "Explanation": "An Explanation relation pertains to one utterance offering an explanation or clarification in response to a question or confusion expressed in another utterance. It aids in providing clarity.",
            "Conditional": "A Conditional relation implies that one utterance presents a condition or hypothetical scenario related to the content of another utterance. It often involves 'if-then' statements.",
            "QAP": "A Question-Answer Pair relation indicates that one utterance contains a question, and another utterance immediately follows with an answer to that question. It demonstrates a direct question-and-answer interaction.",
            "Alternation": "An Alternation relation shows that two utterances present alternative options or choices. It is used when discussing multiple possibilities or courses of action.",
            "Q-Elab": "A Q-Elab relation signifies that one utterance asks a question, and another utterance follows with an elaboration or further explanation of the question or its context.",
            "Result": "A Result relation indicates that one utterance discusses the outcome or consequence of the content presented in another utterance. It shows a cause-and-effect relation.",
            "Background": "In a Background relation, one utterance provides background information or context that is relevant to the content of another utterance. It helps set the stage for the discussion.",
            "Narration": "A Narration relation holds when the main eventualities of two utterances occur in sequence.",
            "Correction": "A Correction relation shows that one utterance corrects or revises the content of another utterance. It is used to rectify errors or inaccuracies.",
            "Parallel": "A Parallel relation relations occur when two or more utterances share similar or related content, often in a parallel or analogous manner. It emphasizes similarities or comparisons.",
            "Contrast": "A Contrast relation signifies that one utterance presents content that is in contrast or opposition to the content of another utterance. It highlights differences or contradictions in the conversation."
        }
        """

        interpersonal_selector = InterpersonalSelector(data, dialogue_id, 5)

        generator = LLMGenerator(stac_relations, edu_to_replace, constraints, selected_examples, interpersonal_selector,"llama-3.3-70b-instruct")

        #generator = LLMGenerator(molweni_relations, edu_to_replace, constraints, selected_examples, interpersonal_selector,"llama-3.3-70b-instruct")

        generated_edus_refined = generator.edu_generation()

        original_dialogue = copy.deepcopy(data[dialogue_id])

        dialogue_refined = support.update_edu(copy.deepcopy(original_dialogue), speaker_to_be_replaced,
                                              replacement_speaker_name, generated_edus_refined)

        if dialogue_refined is None:
            print("Dialogue", i, "not done")
            continue

        new_edus_refined = support.format_edus(generated_edus_refined, replacement_speaker_name)
        updated_json_refined = support.replace_speaker_in_json(copy.deepcopy(original_dialogue),
                                                              speaker_to_be_replaced, new_edus_refined)
        updated_json_refined['selected_speaker'] = speaker_to_be_replaced
        updated_json_refined['replacement_speaker'] = replacement_speaker_name

        synthetic_corpus_refined.append(updated_json_refined)
    except Exception as e:
        print(f"Dialogue {i} not done: {type(e).__name__}: {e}")

with open('augmented_mimic.json', 'w') as json_file:
    json.dump(synthetic_corpus_refined, json_file, indent=4)