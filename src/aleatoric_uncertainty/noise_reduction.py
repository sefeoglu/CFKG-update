## pid to relation type mapping for wikidata verified relations on fewrel

import json
import os
import numpy as np
from tqdm import tqdm
import argparse


def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data 
def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def prompt_generation(sentence, relations, head, tail):

    prompt = f"Sentence: {sentence}\nWhat is the relation between {head} and {tail}  according to given relation types below in the sentence? \nRelation types: {relations}\nAnswer:"
    return prompt

def regenerate_prompts(input_data, relation_folder, out_path):

    for run_id in range(1, 6):
        for task_id in range(1, 9):
            path = f"{relation_folder}/run{run_id}/task{task_id}.json"
            out_file= f"{out_path}/run{run_id}/task{task_id}/test.json"
            task_relations = read_json(path)
            task_prompts = []
            for relation_item in task_relations:
                for item in input_data:
                    relation = item['predicted_pid_relation_type'].replace(" ", "_")
                    if relation == relation_item:
                        sentence = ' '.join(item['tokens'])
                        head = item['h'][0]
                        tail = item['t'][0]
                        prompt = prompt_generation(sentence, ", ".join(task_relations), head, tail)
                        row = {'prompt': prompt, 'relation': relation}
                        task_prompts.append(row)

            write_json(task_prompts, out_file)
            print(f"Generated prompts for run {run_id}, task {task_id}")

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

    return sentence, head_entity, tail_entity

def find_items(wiki_data, sentence, head, tail):

    for item in wiki_data:
        tokens = item['tokens']
        sentence = sentence.replace('\n','')
        joined_sentence = ' '.join(tokens).replace('\n','')

        if sentence.lower() == joined_sentence.lower():
                return item
    print("No match found for: {} - {} - {}".format(sentence, head, tail)) 
    return False



def reannotate_fewrel_testset(fewrel_data_folder, wikidata_path, out_file):

    corrected_data = []
    wikidata_data = read_json(wikidata_path)

    for run_id in range(1, 2):
        for task_id in range(1, 9):
            path_old = f"{fewrel_data_folder}/run{run_id}/task{task_id}/test.json"
            fewrel_data = read_json(path_old)
            for item in tqdm(fewrel_data, desc="Processing fewrel data"):
                sentence_old, head, tail = get_entites_sentence(item['prompt'])
                founded_item = find_items(wikidata_data, sentence_old, head, tail)
                # Do something with the items
                if founded_item:
                    corrected_data.append(founded_item)
    corrected_data = list(set(corrected_data))
    write_json(corrected_data, out_file)



def reduce_noise_bulk_test(fewrel_data_folder, wikidata_data, relation_type, out_file):
    wikidata_data = read_json(wikidata_data)
    mismatch = 0
    relation_type = read_json(relation_type)
    mismatch_list = []
    non_exist = []
    buggy = 0
    wrong_direction = 0
    for item in tqdm(wikidata_data, desc="Processing fewrel data"):
        # print(item)
        # print(item.keys())
        
        if item['has_relation']:
            
            if not item['r_pid'] in item['possible_probs']:
                try:
                    # print("{} - {} ".format(item['possible_probs'][0], item['r_pid']))
                    mismatch += 1
                    item['predicted_pid_relation_type'] = relation_type[item['possible_probs'][0]][0]
                    item['r_pid_relation_type'] = relation_type[item['possible_probs'][0]][0]
                    mismatch_list.append(item)
                except:
                    print("Key error for relation type mapping")
                    non_exist.append(item)
            else:
                try:
                    # print("Match: {} - {} ".format(item['possible_probs'][0], item['r_pid']))
                    if item['possible_probs'][0] == item['r_pid']:
                        
                        item['predicted_pid_relation_type'] = relation_type[item['possible_probs'][0]][0]
                        item['r_pid_relation_type'] = relation_type[item['r_pid']][0]
                        mismatch_list.append(item)
                    else:
                        print("Wrong direction: Predicted: {} - Actual: {}".format(relation_type[item['possible_probs'][0]][0], relation_type[item['r_pid']][0]))
                        item['predicted_pid_relation_type'] = relation_type[item['possible_probs'][0]][0]
                        item['r_pid_relation_type'] = relation_type[item['possible_probs'][0]][0]
                        mismatch_list.append(item)
                        wrong_direction += 1
                        

                except:
                    buggy += 1
                    # print("Key error for relation type mapping")
                
        else:
            # print(not item['has_relation'])
            # print(item)
            # mismatch += 1
            item['predicted_pid_relation_type'] = relation_type[item['r_pid']][0]
            item['r_pid_relation_type'] = relation_type[item['r_pid']][0]
            mismatch_list.append(item)


    write_json(non_exist, out_file.replace(".json","_non_exist.json"))
    write_json(mismatch_list, out_file)
        # break
    print("Mismatched count: {}".format(mismatch))
    print("Total mismatches: {}".format(len(mismatch_list)))
    print("Buggy count: {}".format(buggy))
    print("Wrong direction count: {}".format(wrong_direction))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine contextual information from multiple sources.")
    parser.add_argument("--input_folder", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/canonical_relation_maps/fewrel/corrected_test_data.json", help="Path to the input JSON file containing contextual information.")
    parser.add_argument("--wikidata_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/canonical_relation_maps/fewrel/fewrel_wikidata_relation_mapping.json", help="Path to the input JSON file containing contextual information.")
    parser.add_argument("--relation_type", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/canonical_relation_maps/fewrel/pid2name.json", help="Path to the config JSON file.")
    parser.add_argument("--output_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/fewrel_corrected_results/fewrel_revisited/test/", help="Path to the output JSON file to save combined information.")
    parser.add_argument("--relation_folder", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/fewrel_corrected_results/fewrel_revisited/relations", help="Path to the output JSON file to save combined information.")
    args = parser.parse_args()
    input_folder = args.input_folder
    wikidata_file = args.wikidata_file
    relation_type = args.relation_type
    out_path = args.output_file
    relation_folder = args.relation_folder
    input_data = read_json(input_folder)
    # reduce_noise_bulk_test(fewrel_folder, wikidata_file, relation_type, out_path)
    # reannotate_fewrel_testset(fewrel_folder, wikidata_file, out_path)
    
    regenerate_prompts(input_data, relation_folder, out_path)

   
