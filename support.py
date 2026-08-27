import json, copy, re

def replace_speaker_in_json(dialogue, old_speaker, new_edus):
    edu_index = 0
    new_dialogue = copy.deepcopy(dialogue)

    pattern = re.compile(re.escape(old_speaker), re.IGNORECASE)

    new_speaker = new_edus[0]["speaker"]

    for edu in new_dialogue["edus"]:
        if edu["speaker"] == old_speaker and edu_index < len(new_edus):
            edu["speaker"] = new_speaker
            edu["text"] = new_edus[edu_index]["text"]
            edu_index += 1

        edu["text"] = pattern.sub(new_speaker, edu["text"])

    return new_dialogue

def update_edu(dialogue, old_speaker, new_speaker, new_edus):
    speaker_indices = [i for i, edu in enumerate(dialogue['edus']) if edu['speaker'] == old_speaker]

    if len(speaker_indices) != len(new_edus):
        print(f"Error: the number of new EDU does not match the number of EDU for {old_speaker}.")
        return None

    for i, idx in enumerate(speaker_indices):
        dialogue['edus'][idx]['text'] = new_edus[i] 
        dialogue['edus'][idx]['speaker'] = new_speaker

    return dialogue

def format_edus(edu_texts, new_speaker):
    return [{"speaker": new_speaker, "text": text} for text in edu_texts]

def open_corpus(corpus_path):
    with (open(f'../corpora/{corpus_path}.json', 'r') as file):
        data = json.load(file)

    return data