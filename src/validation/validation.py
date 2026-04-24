from shacl_validation import SHACLValidation
from typing import Any, Optional
import os
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


class Validation:
    def __init__(self, inverse_shacl_file: str, canonical_shacl_file: str) -> None:
        self.inverse_shacl_validator = SHACLValidation(inverse_shacl_file)
        self.canonical_shacl_validator = SHACLValidation(canonical_shacl_file)

    def validate(self, predefined_set: Any, prediction: Any) -> Optional[Any]:
        return self.constraints_check(predefined_set, prediction)

    def constraints_check(self, predefined_set: Any, prediction: Any) -> Optional[Any]:
        canonical_suggestion = self.canonical_constraints_check(prediction)
        if canonical_suggestion is not None:
            return {"suggestion": canonical_suggestion, "message": "Canonical"}

        inverse_suggestion = self.inverse_constraints_check(prediction)

        if inverse_suggestion is not None:
            return {"suggestion": inverse_suggestion, "message": "Inverse"}
        return {"suggestion": None, "message": "No_suggestion"}

    def inverse_constraints_check(self, prediction: Any) -> Optional[Any]:
        return self.inverse_shacl_validator.check_constraints(prediction)

    def canonical_constraints_check(self, prediction: Any) -> Optional[Any]:
        return self.canonical_shacl_validator.check_constraints(prediction)

    def relation_mapping_check(self, predefined_set: Any, prediction: Any) -> Optional[Any]:
        return None

def apply_only_shacl_validation(inverse_shacl_file: str, canonical_shacl_file: str, prediction_folder: Any, ground_truth_folder: Any, predefined_set_folder: Any) -> Optional[Any]:
    
    
    validator = Validation(inverse_shacl_file, canonical_shacl_file)
    validation_results = []
    for run_id in range(1,6):
        gt_relations_set = []
        predefined_relations = []
        for task_id in range(1,9):
            predefined_set_file = f"{predefined_set_folder}/run{run_id}/task{task_id}.json"
            predefined_set = read_json(predefined_set_file)
            predefined_relations.extend(predefined_set)
            ground_truth_file = f"{ground_truth_folder}/run{run_id}/task{task_id}/test.json"
            prediction_file = f"{prediction_folder}/run{run_id}/test_pred_{task_id}.json"
            predictions = read_json(prediction_file)
            ground_truths = read_json(ground_truth_file)
            gt_relations = [item['relation'] for item in ground_truths]
            gt_relations_set.extend(gt_relations)
            print(len(gt_relations_set))
            print(len(predictions))
            print(predictions[0])
            print(gt_relations_set[0])
    
            for index, pred in enumerate(predictions):
                predefined_set ="followed_by"  # Example predefined set
                prediction = pred['predict']
                # print(f"Evaluating Run {run_id}, Task {task_id}, Instance {index+1}: Prediction = {prediction}, Ground Truth = {gt_relations[index]}")
                suggestions = validator.validate(predefined_set, prediction)
                # pred['suggestions'] = suggestions
                print(f"Run:    {run_id}, Task: {task_id}, Instance: {index+1}")
                if prediction != gt_relations_set[index]:
                    if suggestions['suggestion'] is not None:

                        #print(f"Suggestion made: {suggestions['suggestion']} due to {suggestions['message']}")
                        row = {
                            "run_id": run_id,
                            "task_id": task_id,
                            "prediction": prediction,
                            "suggestions": suggestions['suggestion'],
                            "message": suggestions['message'],
                            "gt_relation": gt_relations_set[index],
                            "index": index
                        }
                        
                    else:
                        if prediction in predefined_relations:
                            row = {
                                "run_id": run_id,
                                "task_id": task_id,
                                "prediction": prediction,
                                "suggestions": "None",
                                "message": "Incorrect_Prediction_In_Predefined_Set_No_Suggestion",
                                "gt_relation": gt_relations_set[index],
                                "index": index
                            }
                        else:
                            row = {
                                "run_id": run_id,
                                "task_id": task_id,
                                "prediction": prediction,
                                "suggestions": "None",
                            "message": "Hallucinatted_Relation_No_Suggestion",
                            "gt_relation": gt_relations_set[index],
                            "index": index
                        }
                
                if prediction == gt_relations_set[index]:
                    row = {
                        "run_id": run_id,
                        "task_id": task_id,
                        "prediction": prediction,
                        "suggestions": "None",
                        "message": "Correct_Prediction",
                        "gt_relation": gt_relations_set[index],
                        "index": index
                    }
                validation_results.append(row)
     

    write_json(validation_results, f"{prediction_folder}/canonical_errors/validation_results.json")



def apply_full_validation_pipeline():
    pass  # Placeholder for the full validation pipeline implementation


if __name__ == "__main__":
    inverse_shacl_file = "./shapes/inverse_relations_shapes.ttl"
    canonical_shacl_file = "./shapes/canonical_relations_shapes.ttl"
    ground_truth_file = "/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/fewrel/test"
   
    prediction_folder_list = {
        'ewc':"/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/tmlr/ewc",
        'mas':"/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/tmlr/mas",
        'baseline':"/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/tmlr/baseline",
        'si':"/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/tmlr/si"
    }
    predefined_set_folder = "/Users/sefika/phd_projects/llm-catastrophic-re/results_all/fewrel_corrected_results/fewrel/relations"
    print("Applying SHACL-only validation for EWC predictions...")
    apply_only_shacl_validation(inverse_shacl_file, canonical_shacl_file, prediction_folder_list['ewc'], ground_truth_file, predefined_set_folder)
    
    print("Applying SHACL-only validation for MAS predictions...")
    apply_only_shacl_validation(inverse_shacl_file, canonical_shacl_file, prediction_folder_list['mas'], ground_truth_file, predefined_set_folder)
    print("Applying SHACL-only validation for Baseline predictions...")
    apply_only_shacl_validation(inverse_shacl_file, canonical_shacl_file, prediction_folder_list['baseline'], ground_truth_file, predefined_set_folder)
    print("Applying SHACL-only validation for SI predictions...")   
    apply_only_shacl_validation(inverse_shacl_file, canonical_shacl_file, prediction_folder_list['si'], ground_truth_file, predefined_set_folder)
    print("Validation completed.")
    
    