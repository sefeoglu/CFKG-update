
import json
import os

import sys
import argparse 



def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_human_judged_data(predictions):
    human_judged_data = []
    for pred in predictions:
        if isinstance(pred, list):
            pred = [p.strip() for p in pred if p != '']
            print( pred)
            if len(pred) < 3:
                human_judged_data.append({'suggestion': '', 'score': '', 'human_judged': False})
                continue
            suggestion = pred[2]
            score = pred[0].split(' ')[-1].strip()
            # explaination = pred[2]
            if "no human" in suggestion.lower():
                human_judged_data.append({'suggestion': suggestion, 'score': score,  'human_judged': False})
            else:
                human_judged_data.append({'suggestion': suggestion, 'score': score, 'human_judged': True})
        else:
            human_judged_data.append({'suggestion': '', 'score': '', 'human_judged': False})
    return human_judged_data

def evaluate_with_llm_as_judge(predictions, human_judgements):
    known_count = 0
    false_count = 0
    for i, pred in enumerate(predictions):

        if pred['truth'] != pred['predict'] and human_judgements[i]['human_judged']:
            known_count += 1
            
        # elif pred['truth'] == pred['predict'] and human_judgements[i]['human_judged']:
        #     known_count += 1

        if pred['truth'] != pred['predict']:
            false_count += 1

    return {'total_human_judged': len(human_judgements), 'correctly_judged': known_count, 'false_positives': false_count}

def main(args):
    results_list = []
    for run_id in range(1, 6):
        for task_id in range(8,9):
            prediction_file_path = args.predictions + f"/model_{run_id}/"
            prediction_data = []
            for file in os.listdir(prediction_file_path):
                if file.startswith(f"test_pred_{task_id}_prompts") and file.endswith(".json"):
                    file_path = os.path.join(prediction_file_path, file)
                    print(f"Found file: {file_path}")
            
                    pred_data = read_json(file_path)
                    prediction_data.extend(pred_data)
            preds = [item['predictions_rationale'] for item in prediction_data]

            human_judged_data = get_human_judged_data(preds)
            llm_judged_data   = evaluate_with_llm_as_judge(prediction_data, human_judged_data)
            result = {'run_id': run_id, 'task_id': task_id, 'human_judged': human_judged_data, 'llm_judged': llm_judged_data}
            results_list.append(result)

    write_json(results_list, args.predictions + args.output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate predictions using LLM as a judge.")
    parser.add_argument("--predictions", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/results_all/results_rationale_nli/gpt_nli/tacred_mas_top_prob_1", help="Path to the JSON file containing predictions.")
    parser.add_argument("--output", type=str, default="llm_judged_results.json", help="Path to the output JSON file.")
    args = parser.parse_args()
    main(args)

