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

def get_task_selection(relation_names, num_tasks=5):
    """
    Selects tasks based on the provided relation names.

    Args:
        relation_names (_type_): _description_
        num_tasks (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    run_tasks = []
    for run_id in range(1, 6):
        task_relations = []
        random.shuffle(relation_names)
        first_tasks = random.sample(relation_names, 6)
        task_relations.append({"task_1": first_tasks})
        remain_relations = list(set(relation_names) - set(first_tasks))
        for i in range(2, num_tasks + 1):
            other_tasks = random.sample(remain_relations, 3)
            task_relations.append({f"task_{i}": other_tasks})
            remain_relations = list(set(remain_relations) - set(other_tasks))
        run_tasks.append({"run_id": run_id, "task_relations": task_relations})
    return run_tasks

def main(path_to_types, path_to_output=None):
    """
    Task selection for the finance dataset in Icremental Learning setting.
    Args:
        path_to_types (_type_): _description_
        path_to_output (_type_, optional): _description_. Defaults to None.
    """
    types_ = read_json(path_to_types)
    relations = types_['relations']
    relation_names = list(relations.keys())

    print(f"Number of relations: {len(relation_names)}")

    task_selection = get_task_selection(relation_names)
    write_json(task_selection, path_to_output)

    # print(f"Task selection written to {path_to_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task selection for finance dataset in Incremental Learning setting.")
    parser.add_argument("--path_to_types", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/fire_types.json", help="Path to the types JSON file.")
    parser.add_argument("--path_to_output", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/tasks.json", help="Path to save the task selection JSON file.")
    
    args = parser.parse_args()
    
    if args.path_to_output is None:
        args.path_to_output = os.path.join(os.path.dirname(args.path_to_types), "task_selection.json")
    
    main(args.path_to_types, args.path_to_output)
 