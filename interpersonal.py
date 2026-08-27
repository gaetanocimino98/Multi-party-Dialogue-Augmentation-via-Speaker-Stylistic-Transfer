# The InterpersonalSelector class extracts discourse interactions of
# speakers identified as exhibiting interpersonal behavior. It scans
# the annotated discourse structure (EDUs and relations) to identify links
# where the left or right EDU is authored by a speaker in the interpersonal
# sample. For each such match, it generates a templated description of the
# relation, which serves as in-context examples to guide LLM-based EDU 
# generation. The final result is a curated set of sampled interactions to 
# prompt LLMs in stylistic rephrasing tasks with rhetorical fidelity.

import random

class InterpersonalSelector:

    def __init__(self, data, dialogue, in_context_examples=5):
        self.data = data
        self.dialogue = dialogue
        self.in_context_examples = in_context_examples

    def create_interpersonal_dictionary(self, interpersonal_speakers_samples):

        interpersonal_sentences = {}

        for i in range(len(self.data)):
            if i == self.dialogue:
                continue

            dialogue = self.data[i]
            edus = dialogue['edus']
            relations = dialogue['relations']

            for j in range(len(relations)):
                relation = relations[j]

                rel_type = relation['type']
                left_side = relation['x']
                right_side = relation['y']

                if edus[left_side]['speaker'] in interpersonal_speakers_samples:
                    samples = interpersonal_speakers_samples[edus[left_side]['speaker']]
                    for sample in samples:
                        if sample[0] == rel_type and sample[1] == 'left_side':
                            article = "an" if rel_type[0].lower() in "aeiou" else "a"
                            edu_1 = edus[left_side]['text']
                            edu_2 = edus[right_side]['text']
                            speaker = edus[left_side]['speaker']
                            interpersonal_sentences.setdefault(speaker, []).append(
                                f'The utterance "{edu_1}" '
                                f'by speaker '
                                f'{speaker} is in {article} '
                                f'"{rel_type}" relation with the utterance "{edu_2}".')

                if edus[right_side]['speaker'] in interpersonal_speakers_samples and edus[right_side]['speaker']:
                    samples = interpersonal_speakers_samples[edus[right_side]['speaker']]
                    for sample in samples:
                        if sample[0] == rel_type and sample[1] == 'right_side':
                            article = "an" if rel_type[0].lower() in "aeiou" else "a"
                            edu_1 = edus[left_side]['text']
                            edu_2 = edus[right_side]['text']
                            speaker = edus[right_side]['speaker']
                            interpersonal_sentences.setdefault(speaker, []).append(f'The utterance "{edu_1}" '
                                                           f'is in {article} '
                                                           f'"{rel_type}" relation with the utterance '
                                                           f'"{edu_2}" by speaker {speaker}.')

        all_selected = []

        for values in interpersonal_sentences.values():
            if values:
                num_samples = min(self.in_context_examples, len(values))
                all_selected.extend(random.sample(values, num_samples))

        if len(all_selected) == 0:
            return "Empty"
        else:
            return '\n'.join(all_selected)