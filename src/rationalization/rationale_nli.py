from tqdm import tqdm

import os
import sys
import argparse
import json


def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def covert_logp_to_prob(logp):
    return round(10 ** logp, 6)

def get_entites_sentence(prompt):
    
    sentence = prompt.split('What is the relation')[0].strip()
    
    head_entity = ""
    tail_entity = ""
    sentence = sentence.replace("Sentence:", "").strip()
    # print(prompt)
    query = prompt.split('What is the relation type')[1].strip().split('in the following sentence?')[0].strip()

    entities_query = query.split('between')[1].strip().split('according')[0].strip()
    head_entity = entities_query.split('and')[0].strip()
    tail_entity = entities_query.split('and')[-1].strip()
    relation_type = prompt.split('Relation types:')[-1].strip()

    return sentence, head_entity, tail_entity, relation_type

def get_templates(sentence: str, head_entity: str, tail_entity: str, candidate_relation_type_and_probability: str, prediction: str, relations: str) -> str:

    template = f"""
    You are an expert in evaluating the uncertainty and reliability of rationales provided for candidate relation types between head and tail entities in a sentence.
    Sentence: {sentence}
    Head Entity: {head_entity}
    Tail Entity: {tail_entity}
    [Hypothesis Candidate Relation Types and Probabilities]:
    {candidate_relation_type_and_probability}
    [Premise Possible Relation Types]: {relations}
    [GUIDELINES]
    Based on the provided information, please assess the uncertainty and reliability of the rationale in supporting the candidate relation type between the head and tail entities. 
    Provide a detailed explanation of your assessment.
    Your evaluation should consider the following aspects:
    1. Score: the confidence level of the relation types in supporting the candidate relation types along with probes (scale of 1 to 10, where 1 is very uncertain and 10 is very certain).
    2. Explanation: Why is the relation type between {head_entity} and {tail_entity} is {prediction} according to probabilities?
    3. Suggestions: human suggestion needed or not.
    4.  Classify as :
        Entailment: Hypothesis logically follows from Premise,
        Contradiction: Hypothesis conflicts with Premise, or
        Neutral: Hypothesis is possible but not guaranteed by Premise
    """
    return template

def generate_rationale_prompt(predictions) -> str:
    """
    Generate a prompt for evaluating the rationale of a candidate relation type between head and tail entities in a sentence.

    Args:
        sentence (str): The sentence containing the head and tail entities.
        head_entity (str): The head entity in the sentence.
        tail_entity (str): The tail entity in the sentence.
        candidate_relation_type_and_probability (str): The candidate relation type and its associated probability.
        rationale (str): The rationale provided for the candidate relation type.

    Returns:
        str: The generated prompt for evaluation.
    """
    updated_preds = []
    for prediction in tqdm(predictions, desc="Generating prompts"):
        prompt = prediction['prompt']
        
        preds = prediction['preds']
        relation_prediction = prediction['predict']
        candidate_relation_type_and_probability = {}

        for item in preds:
            if item[0] in candidate_relation_type_and_probability.keys():
                candidate_relation_type_and_probability[item[0]] += covert_logp_to_prob(item[1])
            else:
                candidate_relation_type_and_probability[item[0]] = covert_logp_to_prob(item[1])

        candidates = [ key + " : "+str(value) for key, value in candidate_relation_type_and_probability.items()]
        candidates = "\n".join(candidates)


        sentence, head_entity, tail_entity, relations = get_entites_sentence(prompt)
        prediction['rationale_prompt'] = get_templates(sentence, head_entity, tail_entity, candidates, relation_prediction, relations)
        updated_preds.append(prediction)
    return updated_preds

if __name__ == "__main__":
    argparse.ArgumentParser(description="Generate rationale prompts for relation extraction.")
    parser = argparse.ArgumentParser(description="Generate rationale prompts for relation extraction.")
    parser.add_argument("--input_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/fewrel_corrected_results/tmlr/fewrel_mas_top_prob_5/model_5/test_pred_1.json", help="Path to the input file containing predictions.")
    parser.add_argument("--output_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/rationale_nli/fewrel_mas_top_prob_5/model_5/test_pred_1.json", help="Path to the output file to save generated prompts.")
    args = parser.parse_args()
    for pred_id in range(1, 9):
        print(f"Processing prediction set {pred_id}")   
        input_file = args.input_file.replace(f"test_pred_1.json", f"test_pred_{pred_id}.json")
        output_file = args.output_file.replace(f"test_pred_1.json", f"test_pred_{pred_id}_prompts.json")
        predictions = read_json(input_file)
        print(f"Generating prompts for {input_file}, saving to {output_file}")

        prompts = generate_rationale_prompt(predictions)
        # print("test")
        # print(prompts)
        write_json(prompts, output_file)
