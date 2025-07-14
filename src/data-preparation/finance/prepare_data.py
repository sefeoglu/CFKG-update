
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
def prepare_sentence(data_path):
    """
    Prepares sentences from the data file and writes them to the output path.
    
    Args:
        data_path (str): Path to the input data file.
        output_path (str): Path to the output file where sentences will be written.
    """
    data = read_json(data_path)
    sentences = []
    print(f"Number of data items: {len(data)}")
    for item in data:
        # print(item)
        entites = item['entities']
        relations = item['relations']
        tokens = item['tokens']
    
        for relation_info in relations:
            # print(relation_info)
            head = relation_info['head']
            tail = relation_info['tail']
            relation = relation_info['type']
            head_entity = entites[head]
            tail_entity = entites[tail]
            record = {
                "head": head_entity['text'],
                "tail": tail_entity['text'],
                "relation": relation,
                "sentence": " ".join(tokens),
                "tokens": tokens,
                "head_start": head_entity['start'],
                "head_end": head_entity['end'],
                "tail_start": tail_entity['start'],
                "tail_end": tail_entity['end']
            }
            sentences.append(record)
    print(f"Number of sentences prepared: {len(sentences)}")


        
    return sentences
    
    
def main(path_to_data, path_to_output=None):
    """
    Prepares sentences from the finance dataset.
    
    Args:
        path_to_data (str): Path to the input data file.
        path_to_output (str, optional): Path to the output file where sentences will be written. Defaults to None.
    """
    sentences = prepare_sentence(path_to_data)
    write_json(sentences, path_to_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sentences from the finance dataset.")
    parser.add_argument("--train_path", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/fire_train.json", help="Path to the training data file.")
    parser.add_argument("--test_path", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/fire_test.json", help="Path to the test data file.")
    parser.add_argument("--dev_path", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/fire_dev.json", help="Path to the development data file.")
    parser.add_argument("--output_folder_path", type=str, default="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/prepared_data", help="Path to the output folder where sentences will be written.")
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    dev_path = args.dev_path

    output_folder_path = args.output_folder_path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    main(train_path, os.path.join(output_folder_path, "train_sentences.json"))
    main(test_path, os.path.join(output_folder_path, "test_sentences.json"))
    main(dev_path, os.path.join(output_folder_path, "dev_sentences.json"))

