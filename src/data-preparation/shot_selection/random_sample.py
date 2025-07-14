# random sample selection for active learning in RLHF
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

def random_sample_selection(data, n_samples):
    """
    Randomly selects n_samples from the data.
    
    Args:
        data (list): List of data points to sample from.
        n_samples (int): Number of samples to select.
    
    Returns:
        list: Randomly selected samples.
    """
    relations = [item['relation'] for item in data ]
    relation_types = set(relations)
    samples = []
    for relation in relation_types:
        relation_data = [item for item in data if item['relation'] == relation]
        if len(relation_data) < n_samples:
            samples.extend(relation_data)
        else:
            selected_samples = random.sample(relation_data, n_samples)
            samples.extend(selected_samples)

    return samples

def main():
    parser = argparse.ArgumentParser(description="Random Sample Selection for Active Learning in RLHF")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the selected samples.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    # Read data from input file
    data = read_json(args.input_file)
    
    # Randomly select samples
    selected_samples = random_sample_selection(data, args.n_samples)
    
    # Write selected samples to output file
    write_json(selected_samples, args.output_file)
    # dev dataset
    dev_input_file = args.input_file.replace('train.json', 'dev.json')
    dev_data = read_json(dev_input_file)
    write_json(dev_data, args.output_file.replace('train.json', 'dev.json'))
if __name__ == "__main__":
    main()