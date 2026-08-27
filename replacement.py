# The SpeakerChanger class supports speaker substitution and context selection 
# in the MIMIC framework by modeling the relational behavior of speakers across 
# dialogues. It enables selecting an appropriate replacement speaker (B) to 
# stylistically rephrase utterances from a target speaker (A) in a specific dialogue. 
#
# Main Responsibilities:
#
# 1. Speaker Dictionary Construction:
#    - Aggregates discourse relation instances (EDU-EDU links) for each speaker 
#      across all dialogues, preserving role (initiator/responder/both), relation 
#      type, and interaction partner. This structure allows querying for similar 
#      discourse behavior.
#
# 2. Target Speaker History Extraction:
#    - Filters the speaker dictionary to retain only relations involving the 
#      specified target speaker (A) in the target dialogue. These will be the 
#      utterances selected for rephrasing and used to define constraints.
#
# 3. EDU Constraint Generation:
#    - Identifies EDUs authored by speaker A that participate in discourse relations.
#      Each EDU is labeled with its counterpart, the type of relation, the role 
#      (left-initiator/right-responder/both), and the name of the interaction partner. 
#      These constraints will later guide the generation and validation process.
#
# 4. Speaker Ranking:
#    - Compares the discourse profile of each candidate speaker to that of A 
#      using role-relation pair overlaps. This implements the MIRROR mechanism 
#      from MIMIC: speakers with similar rhetorical behavior are ranked higher 
#      as suitable stylistic substitutes.
#
# 5. Substitute Speaker Selection:
#    - From the MIRROR-based ranking, selects a speaker that satisfies the highest 
#      number of relation-role overlaps with the target speaker. In-context examples 
#      from this substitute are grouped and returned to serve as instances 
#      for LLM prompt construction.
#
# Output:
#    - A randomly selected substitute speaker, their relevant discourse history grouped by 
#      (relation, role), and the set of relation types matched.
#
# This class realizes the functionality of MIRROR, contributing to MIMIC's augmentation 
# process for discourse parsing.

from collections import defaultdict
import random

class SpeakerChanger:

    def __init__(self, data, dialogue, speaker_name, speakers_to_remove, in_context_examples=5):
        self.data = data
        self.in_context_examples = in_context_examples
        self.speaker_names = []
        self.dialogue = dialogue
        self.speaker_to_remove = speakers_to_remove
        self.speaker_name = speaker_name
        self.speak_dict = self.__create_speaker_dictionary()
        self.specific_speak_history = [event for event in self.speak_dict[self.speaker_name] if event[0]["id_dialogue"] == self.dialogue]

        self.__remove_speakers_by_names(speakers_to_remove)

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

                if edus[left_side]['speaker'] not in self.speaker_names and edus[left_side]['speaker'] not in self.speaker_to_remove:
                    self.speaker_names.append(edus[left_side]['speaker'])

                if edus[right_side]['speaker'] not in self.speaker_names and edus[right_side]['speaker'] not in self.speaker_to_remove:
                    self.speaker_names.append(edus[right_side]['speaker'])

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
                          "id_relation": j,
                          "interaction_speaker_name": edus[right_side]['speaker']
                          }])

                    speakers[edus[right_side]['speaker']].append(
                        [{"edu1": edus[left_side]['text'],
                          "edu2": edus[right_side]['text'],
                          "rel_type": rel_type,
                          "role": 'right_side',
                          "id_dialogue": i,
                          "id_speechturn_left": left_side,
                          "id_speechturn_right": right_side,
                          "id_relation": j,
                          "interaction_speaker_name": edus[left_side]['speaker']
                          }])

        return speakers

    def __remove_speakers_by_names(self, speakers_to_remove):
        for speaker in speakers_to_remove:
            if speaker in self.speak_dict:
                del self.speak_dict[speaker]

    def edu_constraints(self):

        edu_to_be_replaced = []
        constraints = defaultdict(list)
        id_speech_turn = []

        for i in range(len(self.specific_speak_history)):
            entry = self.specific_speak_history[i][0]

            left_id = entry['id_speechturn_left']
            right_id = entry['id_speechturn_right']

            if entry['role'] == 'left_side':
                if left_id not in id_speech_turn:
                    edu_to_be_replaced.append((left_id, entry['edu1']))
                    id_speech_turn.append(left_id)
                constraints[left_id].append([entry['edu2'], entry['rel_type'], 'right_side',
                                                entry['interaction_speaker_name']])

            elif entry['role'] == 'right_side':
                if right_id not in id_speech_turn:
                    edu_to_be_replaced.append((right_id, entry['edu2']))
                    id_speech_turn.append(right_id)
                constraints[right_id].append([entry['edu1'], entry['rel_type'], 'left_side',
                                                entry['interaction_speaker_name']])

            else:
                if left_id not in id_speech_turn:
                    edu_to_be_replaced.append((left_id, entry['edu1']))
                    id_speech_turn.append(left_id)

                if right_id not in id_speech_turn:
                    edu_to_be_replaced.append((right_id, entry['edu2']))
                    id_speech_turn.append(right_id)

                constraints[left_id].append([entry['edu2'], entry['rel_type'], 'both-right', 'None'])
                constraints[right_id].append([entry['edu1'], entry['rel_type'], 'both-left', 'None'])

        edus_speaker_sorted = sorted(edu_to_be_replaced, key=lambda x: x[0])

        return edus_speaker_sorted, constraints


    def __speaker_ranking(self):

        results = []

        checks = [[0] * len(self.specific_speak_history) for _ in range(len(self.speaker_names))]

        relations_su = sorted({e[0]['rel_type'] for e in self.specific_speak_history})
        dimensions = [(r, role) for r in relations_su for role in ('left_side', 'right_side')]

        for i in range(len(self.speaker_names)):
            entries = self.speak_dict[self.speaker_names[i]]

            counts = {d: 0 for d in dimensions}

            for j in range(len(entries)):
                entry = entries[j]
                rel_type = entry[0]['rel_type']
                role = entry[0]['role']

                key = (rel_type, role)
                if key in counts:
                    counts[key] += 1

                for x in range(len(self.specific_speak_history)):
                    check = self.specific_speak_history[x][0]
                    target_rel_type = check['rel_type']
                    target_role = check['role']

                    if rel_type == target_rel_type and role == target_role:
                        checks[i][x] = checks[i][x] + 1

            results.append({"Name": self.speaker_names[i],
                            "Checks": checks[i],
                            "Dominance": [counts[d] for d in dimensions]})

        return results

    def __speaker_relation_match(self):

        data = self.__speaker_ranking()

        bin = []

        flag = True

        while flag:
            for entry in data:
                name = entry["Name"]
                checks = entry["Checks"]

                if all(x >= self.in_context_examples for x in checks):
                    bin.append(name)

            if len(bin) > 0:
                flag = False
            else:
                self.in_context_examples = self.in_context_examples - 1
                if self.in_context_examples == 0:
                    flag = False

        if len(bin) == 0:
            best_entry = max(data, key=lambda entry: sum(
                all(entry["Dominance"][i] >= other["Dominance"][i]
                    for i in range(len(entry["Dominance"])))
                for other in data
            ))["Name"]
            bin.append(best_entry)

        return bin

    def select_random_name(self):

        bin = self.__speaker_relation_match()
        random_choice = random.choice(bin)
        entries = self.speak_dict[random_choice]
        in_context_examples = {}

        for j in range(len(entries)):
            entry = entries[j]
            rel_type = entry[0]['rel_type']
            role = entry[0]['role']

            for x in range(len(self.specific_speak_history)):
                check = self.specific_speak_history[x][0]
                target_rel_type = check['rel_type']
                target_role = check['role']

                if rel_type == target_rel_type and role == target_role:
                    key = (target_rel_type, target_role)
                    if key not in in_context_examples:
                        in_context_examples[key] = []
                    in_context_examples[key].append(entry)
                    break

        relation_types = list(set(key[0] for key in in_context_examples.keys()))

        return random_choice, in_context_examples, relation_types, self.in_context_examples








