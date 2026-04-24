# pip install evaluate bert-score
import evaluate
import json
import os
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

import sys
import argparse 
os.getenv("HF_HUB_OFFLINE")


def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def compute_bertscore(predictions, references):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(predictions, references)
    return P, R, F1


def compute_average_bertscore(results):
    avg_p  = sum(results["precision"]) / len(results["precision"])
    avg_r  = sum(results["recall"])    / len(results["recall"])
    avg_f1 = sum(results["f1"])        / len(results["f1"])
    return avg_p, avg_r, avg_f1



def main(args):
    results_list = []
    for run_id in range(1, 6):
        ref_data = []
        for task_id in range(1, 9):
            prediction_file_path = args.predictions + f"/run{run_id}/test_pred_{task_id}.json"
            reference_file_path  = args.references + f"/run{run_id}/task{task_id}/test.json"
            
            pred_data = read_json(prediction_file_path)
            ref_data.extend(read_json(reference_file_path))

            preds = [item['predict'] for item in pred_data]
            refs  = [item['relation'] for item in ref_data]

            P, R, F1 = compute_bertscore(preds, refs)
            results = {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist()
            }
            avg_p, avg_r, avg_f1 = compute_average_bertscore(results)
            result = {'run_id': run_id, 'task_id': task_id, 'avg_precision': avg_p, 'avg_recall': avg_r, 'avg_f1': avg_f1}
            results_list.append(result)
            print(f"Avg BERTScore — P: {avg_p:.4f}  R: {avg_r:.4f}  F1: {avg_f1:.4f}")

    output_path = args.predictions + "/" + args.output
    write_json(results_list, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BERTScore between predictions and references.")
    parser.add_argument("--predictions", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/last_results/tacred/probs/tacred_si_top_prob_5", help="Path to the JSON file containing predictions.")
    parser.add_argument("--references", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/FSCRE/tacred/5way5shot-test", help="Path to the JSON file containing references.")
    parser.add_argument("--output", type=str, default="bertscore_results.json", help="Path to the output JSON file.")
    args = parser.parse_args()



    main(args)
