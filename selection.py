# The SpeakerSelector class implements a speaker ranking mechanism for selecting
# a speaker within a dialogue, based on discourse structure
# and interactional features. This aligns with the MASK component in MIMIC, which 
# identifies key speakers whose utterances will be rephrased.
#
# Main Responsibilities:
#
# 1. Speaker-Relation Indexing:
#    - Builds a mapping from speaker names to their discourse relation history.
#      Each entry captures the two EDUs in the relation, the rhetorical role 
#      (left-initiator/right-responder/both), and their speech turn positions 
#      within a dialogue.
#
# 2. Relation Frequency Scoring:
#    - Computes the relative prominence of a speaker's use of rhetorical relations
#      by comparing their local distribution in the current dialogue against the
#      overall corpus distribution.
#    - The "f_rel" metric rewards speakers who frequently produce diverse or
#      high-frequency relations, reflecting structural centrality.
#
# 3. Discourse Link Tightness:
#    - Computes "f_link", the average distance between paired EDUs in each relation
#      authored by the speaker.
#
# 4. Score Normalization and Aggregation:
#    - Both metrics (f_rel and f_link) are min-max normalized and combined into
#      a unified speaker score. The speaker with the highest score is a suitable
#      candidate for substitution under MIMIC's strategy.
#
# Output:
#    - Returns a ranked list of speakers in the target dialogue, each with a 
#      composite score reflecting their rhetorical centrality and relational
#      connectivity. This score helps identify which speaker is most suitable 
#      for rephrasing in subsequent stages of the MIMIC pipeline.

from collections import defaultdict

class SpeakerSelector:

    def __init__(self, data, dialogue):
        self.data = data
        self.dialogue = dialogue
        self.speak_dict = self.__create_speaker_dictionary()

    def __create_speaker_dictionary(self):

        speakers = defaultdict(list)

        for i in range(len(self.data)):
            dialogue = self.data[i]
            edus = dialogue['edus']
            relations = dialogue['relations']

            for j in range(len(relations)):
                relation = relations[j]

                rel_type = relation['type']
                left_side = relation['x']
                right_side = relation['y']

                if edus[left_side]['speaker'] == edus[right_side]['speaker']:
                    speakers[edus[left_side]['speaker']].append(
                        [{"edu1": edus[left_side]['text'],
                          "edu2": edus[right_side]['text'],
                          "rel_type": rel_type,
                          "role": 'both',
                          "id_speechturn_left": left_side,
                          "id_speechturn_right": right_side,
                          "id_dialogue": i,
                          "id_relation": j
                          }])
                else:
                    speakers[edus[left_side]['speaker']].append(
                        [{"edu1": edus[left_side]['text'],
                         "edu2": edus[right_side]['text'],
                         "rel_type": rel_type,
                         "role": 'left_side',
                         "id_speechturn_left": left_side,
                         "id_speechturn_right": right_side,
                         "id_dialogue": i,
                         "id_relation": j
                        }])

                    speakers[edus[right_side]['speaker']].append(
                        [{"edu1": edus[left_side]['text'],
                         "edu2": edus[right_side]['text'],
                         "rel_type": rel_type,
                         "role": 'right_side',
                         "id_dialogue": i,
                         "id_speechturn_left": left_side,
                         "id_speechturn_right": right_side,
                         "id_relation": j
                         }])

        return speakers

    def __compute_corpus_relations(self):

        global_relations = []
        rel_dict = {}

        for i in range(len(self.data)):
            dialogue = self.data[i]
            relations = dialogue['relations']

            for j in range(len(relations)):
                relation = relations[j]

                if relation['type'] not in global_relations:
                    global_relations.append(relation['type'])
                    rel_dict[relation['type']] = 1
                else:
                    rel_dict[relation['type']] = rel_dict[relation['type']] + 1

        return global_relations, rel_dict

    def __extract_speaker_names(self):

        dialogue = self.data[self.dialogue]

        edus = dialogue['edus']

        return list(set(entry['speaker'] for entry in edus))

    def __compute_speaker_relation_score(self):

        relation_names, corpus_relations = self.__compute_corpus_relations()

        speaker_names = self.__extract_speaker_names()

        f_rel_scores = []

        for s in range(len(speaker_names)):

            speaker_name = speaker_names[s]

            speaker_history = [event for event in self.speak_dict[speaker_name] if event[0]["id_dialogue"] == self.dialogue]

            speaker_relations = {r: 0 for r in relation_names}

            for i in range(len(speaker_history)):
                speaker_relations[speaker_history[i][0]["rel_type"]] += 1

            total_corpus_relations = sum(corpus_relations.values())
            p_D = {r: corpus_relations[r] / total_corpus_relations for r in corpus_relations}

            f_rel = sum(speaker_relations[r] / p_D[r] for r in speaker_relations if p_D[r] > 0)

            total_speaker_relations = sum(speaker_relations.values())
            f_rel_norm = f_rel / total_speaker_relations if total_speaker_relations > 0 else 0

            f_rel_scores.append({"Name": speaker_name, "f_rel_norm": f_rel_norm})

        return f_rel_scores

    def __compute_speaker_link_score(self):

        speaker_names = self.__extract_speaker_names()

        f_link_scores = []

        for s in range(len(speaker_names)):

            speaker_name = speaker_names[s]

            speaker_history = [event for event in self.speak_dict[speaker_name] if event[0]["id_dialogue"] == self.dialogue]

            total_difference = 0
            total_pairs = 0

            for entry in speaker_history:
                edu = entry[0]
                i = edu['id_speechturn_left']
                j = edu['id_speechturn_right']
                total_difference += abs(i - j)
                total_pairs += 1

            f_conn = total_difference / total_pairs if total_pairs > 0 else 0

            f_link_scores.append({"Name": speaker_name, "f_link": f_conn})

        return f_link_scores

    def min_max_normalize(self, values):
        min_val = min(values)
        max_val = max(values)

        if max_val != min_val:
            return [(val - min_val) / (max_val - min_val) for val in values]
        else:
            return [0.5] * len(values)

    def compute_speaker_score(self):

        f_rel_scores = self.__compute_speaker_relation_score()
        f_link_scores = self.__compute_speaker_link_score()

        rel_scores_values = [score['f_rel_norm'] for score in f_rel_scores]
        link_scores_values = [score['f_link'] for score in f_link_scores]

        normalized_rel_scores = self.min_max_normalize(rel_scores_values)
        normalized_link_scores = self.min_max_normalize(link_scores_values)

        final_scores = []
        for i in range(len(f_rel_scores)):
            score = normalized_rel_scores[i] + normalized_link_scores[i]
            final_scores.append({"Name": f_rel_scores[i]['Name'], "score": score})

        return final_scores