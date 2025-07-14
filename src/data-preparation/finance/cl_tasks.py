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

def get_cl_tasks(relation_names, sentences, out_folder, prefix_file='train', num_tasks=5):
    """
    Selects tasks based on the provided relation names.

    Args:
        relation_names (_type_): _description_
        num_tasks (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
   
    for run_id in range(1, 6):
        
        for i in range(1, num_tasks + 1):
            taks_relations = relation_names[0]['run_id'] == run_id and relation_names[0]
            taks_relations = relation_names[0]['task_relations'][i - 1][f'task_{i}']
            task_sentences = []
            for sentence in sentences:
                if sentence['relation'] in taks_relations:
                    task_sentences.append(sentence)
            path_to_output = f"{out_folder}/run_{run_id}/task_{i}/{prefix_file}.json"
            write_json(task_sentences, path_to_output)
                
def main(path_to_types, path_to_sentences, out_folder, prefix_file='train'):
    """
    Task selection for the finance dataset in Icremental Learning setting.
    Args:
        path_to_types (_type_): _description_
        path_to_sentences (_type_): _description_
        out_folder (_type_): _description_
    """
    task_relations = read_json(path_to_types)


    sentences = read_json(path_to_sentences)

    get_cl_tasks(task_relations, sentences, out_folder, prefix_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CL tasks for finance dataset")
    parser.add_argument("--path_to_types", type=str, default='/Users/sefika/phd_projects/tgdk-paper/data/finance/data/tasks.json',  help="Path to the relation types JSON file")
    parser.add_argument("--path_to_sentences", type=str, default='/Users/sefika/phd_projects/tgdk-paper/data/finance/data/prepared_data/test_sentences.json', help="Path to the sentences JSON file")
    parser.add_argument("--out_folder", type=str,default='/Users/sefika/phd_projects/tgdk-paper/data/finance/data/cl_task/test',  help="Output folder to save the tasks")
    parser.add_argument("--prefix_file", type=str, default='test', help="Prefix for the output files")
    args = parser.parse_args()
    
    main(args.path_to_types, args.path_to_sentences, args.out_folder, args.prefix_file)
