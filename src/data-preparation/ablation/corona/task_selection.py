# split 6 relations for the first task and then 3 relations for others.

import os
import json
import random 
import argparse
def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def get_task_selection(relation_names, out_folder, num_tasks=6):
    """
    Selects tasks based on the provided relation names.

    Args:
        relation_names (_type_): _description_
        num_tasks (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
 
    for run_id in range(1, 6):
        random.shuffle(relation_names)
        first_tasks = random.sample(relation_names, 4)
        
        remain_relations = list(set(relation_names) - set(first_tasks))
        write_json(first_tasks, os.path.join(out_folder, f"run_{run_id}/task_1.json"))

        for i in range(2, num_tasks):
            other_tasks = random.sample(remain_relations, 2)

            remain_relations = list(set(remain_relations) - set(other_tasks))
            write_json(other_tasks, os.path.join(out_folder, f"run_{run_id}/task_{i}.json"))
        
            


def main(path_to_types, path_to_output=None):
    """
    Task selection for the finance dataset in Icremental Learning setting.
    Args:
        path_to_types (_type_): _description_
        path_to_output (_type_, optional): _description_. Defaults to None.
    """
    types_ = read_json(path_to_types)
    relation_names = types_

    print(f"Number of relations: {len(relation_names)}")

    get_task_selection(relation_names, out_folder=path_to_output)


    # print(f"Task selection written to {path_to_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task selection for finance dataset in Incremental Learning setting.")
    parser.add_argument("--path_to_types", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/dataset/corona/relations.json", help="Path to the types JSON file.")
    parser.add_argument("--path_to_output", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/dataset/corona/tasks", help="Path to save the task selection JSON file.")
    
    args = parser.parse_args()
    
    if args.path_to_output is None:
        args.path_to_output = os.path.join(os.path.dirname(args.path_to_types), "task_selection.json")
    
    main(args.path_to_types, args.path_to_output)
 