import json
import random

# Sample TACRED label list (you should load the full schema from TACRED)
TACRED_LABELS = [
    "per:place_of_birth", "org:top_members/employees", "per:spouse",
    "org:city_of_headquarters", "per:children", "no_relation"
]

# Template variations
CORRECT_TEMPLATES = [
    "The relation is {label} because it accurately reflects the information in the sentence.",
    "This is an instance of {label}, based on how the subject and object are related.",
    "{label} fits here because the sentence directly indicates that relation."
]

INCORRECT_TEMPLATES = [
    "The relation is {label}, but this does not match the context of the sentence.",
    "Incorrectly identified as {label}; the sentence does not imply this relation.",
    "Although labeled as {label}, the actual meaning of the sentence differs."
]

def highlight(sentence_tokens, subj, obj):
    return sentence_tokens.replace(subj, f"<SUBJ>{subj}</SUBJ>").replace(obj, f"<OBJ>{obj}</OBJ>")

def generate_prompt(tokens, subj, obj):
    sentence = " ".join(tokens)
    sentence = highlight(sentence, subj, obj)
    return f"Identify the relation between the <SUBJ>subject</SUBJ> and <OBJ>object</OBJ> in the sentence: \"{sentence}\""

def generate_chosen(relation):
    template = random.choice(CORRECT_TEMPLATES)
    return template.format(label=relation)

def generate_rejected(relation):
    incorrect = random.choice([lbl for lbl in TACRED_LABELS if lbl != relation])
    template = random.choice(INCORRECT_TEMPLATES)
    return template.format(label=incorrect)

def convert_tacred_to_dpo(tacred_json_path, output_json_path, limit=500):
    with open(tacred_json_path, "r") as f:
        data = json.load(f)

    dpo_data = []
    for ex in data[:limit]:
        prompt = generate_prompt(ex["tokens"], ex["subject"], ex["object"])
        chosen = generate_chosen(ex["relation"])
        rejected = generate_rejected(ex["relation"])
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    with open(output_json_path, "w") as f:
        json.dump(dpo_data, f, indent=2)

    print(f"Converted {len(dpo_data)} TACRED examples to DPO format.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TACRED dataset to DPO format.")
    parser.add_argument("tacred_json_path", type=str, help="Path to the TACRED JSON file.")
    parser.add_argument("output_json_path", type=str, help="Path to save the DPO formatted JSON file.")
    parser.add_argument("--limit", type=int, default=500, help="Number of examples to convert (default: 500).")

    args = parser.parse_args()
    convert_tacred_to_dpo(args.tacred_json_path, args.output_json_path, args.limit)
