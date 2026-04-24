import os
import json
import random 
import argparse
def flan_t5_format(data, relation_types):
    """
    Converts the data to Flan-T5 format.
    
    Args:
        data (list): List of data points to convert.
    
    Returns:
        list: Data in Flan-T5 format.
    """
    formatted_data = []
    
    relations = ", ".join(relation_types)
    for item in data:
        # print(item)
        prompt = f"What is the relation type between {item['e1']['entity']} and {item['e2']['entity']} in the sentence? Sentence: {item['input']} Relation types: {relations}. Answer:"
        formatted_data.append({
            "prompt": prompt,
            "relation": item['relation']
        })
    return formatted_data

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
def get_test_samples(data, relation_names):
    samples = []
    for relation in relation_names:
        for item in data:
            if item['relation'] == relation:
                print(item)
                samples.append(item)
    samples = flan_t5_format(samples, relation_names)
    return samples
def get_relation_samples(data, relation_names):
    samples = []
    for relation in relation_names:
        relation_samples = data[relation]
        samples.extend(relation_samples)
    samples = flan_t5_format(samples, relation_names)
    return samples
def main(path_to_types, path_to_samples, out_folder, prefix_file='train', num_tasks=6):
    """
    Task selection for the finance dataset in Icremental Learning setting.
    Args:
        path_to_types (_type_): _description_
        path_to_sentences (_type_): _description_
        out_folder (_type_): _description_
    """
    data = read_json(path_to_samples)
    cummulative_data = []
    for run_id in range(1, 6):
        
        for task_id in range(1, num_tasks):
            task_relations = read_json(os.path.join(path_to_types, f"run_{run_id}/task_{task_id}.json"))
            if prefix_file == 'test':
                sentences = get_test_samples(data, task_relations)
            else:   
                sentences = get_relation_samples(data, task_relations)
            cummulative_data.extend(sentences)
            path_to_output = f"{out_folder}/run_{run_id}/task_{task_id}/{prefix_file}.json"
            write_json(cummulative_data, path_to_output)
        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CL tasks for finance dataset")
    parser.add_argument("--path_to_types", type=str, default='/Users/sefika/phd_projects/llm-catastrophic-re/dataset/corona/tasks',  help="Path to the relation types JSON file")
    parser.add_argument("--path_to_sentences", type=str, default='/Users/sefika/phd_projects/continual-corona-news-kg/results/annotation/test/test_majority_voting_validated_relations.json', help="Path to the sentences JSON file")
    parser.add_argument("--out_folder", type=str,default='/Users/sefika/phd_projects/llm-catastrophic-re/dataset/corona/test',  help="Output folder to save the tasks")
    parser.add_argument("--prefix_file", type=str, default='test', help="Prefix for the output files")
    args = parser.parse_args()
    
    main(args.path_to_types, args.path_to_sentences, args.out_folder, args.prefix_file)
