import os
import json
import random 
import argparse

random.seed(42)  # For reproducibility
# This script prepares data in different formats for a finance dataset.
def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def llama_format(data):
    """
    Converts the data to Llama format.
    
    Args:
        data (list): List of data points to convert.
    
    Returns:
        list: Data in Llama format.
    """
    formatted_data = []
    relation_types = set(item['relation'] for item in data)
    relation_types = sorted(relation_types)  # Sort for consistency
    relations = ", ".join(relation_types)
    for item in data:
        item['sentence'] = item['sentence'].replace(item['head'], f"<e1>{item['head']}</e1>").replace(item['tail'], f"<e2>{item['tail']}</e2>")
        prompt = f"What is the relation type between two entities in the sentence? {item['sentence']}. Possible relations: {relations}. Answer:"
        # Llama format uses 'prompt' and 'completion'
        formatted_data.append({
            "prompt": prompt,
            "relation": item['relation']
        })
    
    return formatted_data

def mistral_format(data):
    """
    Converts the data to Mistral format.
    
    Args:
        data (list): List of data points to convert.
    
    Returns:
        list: Data in Mistral format.
    """
    formatted_data = []
    relation_types = set(item['relation'] for item in data)
    relation_types = sorted(relation_types)  # Sort for consistency
    relations = ", ".join(relation_types)
    for item in data:
        item['sentence'] = item['sentence'].replace(item['head'], f"<e1>{item['head']}</e1>").replace(item['tail'], f"<e2>{item['tail']}</e2>")
        prompt = f"What is the relation type between two entities in the sentence? {item['sentence']}. Possible relations: {relations}. Answer:"
        # Mistral format uses 'prompt' and 'completion'
        formatted_data.append({
            "prompt": prompt,
            "relation": item['relation']
        })

    return formatted_data

def flan_t5_format(data):
    """
    Converts the data to Flan-T5 format.
    
    Args:
        data (list): List of data points to convert.
    
    Returns:
        list: Data in Flan-T5 format.
    """
    formatted_data = []
    relation_types = set(item['relation'] for item in data)
    relation_types = sorted(relation_types)  # Sort for consistency
    relations = ", ".join(relation_types)
    for item in data:
        item['sentence'] = item['sentence'].replace(item['head'], f"<e1>{item['head']}</e1>").replace(item['tail'], f"<e2>{item['tail']}</e2>")
        prompt = f"What is the relation type between two entities in the sentence? {item['sentence']}. Possible relations: {relations}. Answer:"
        formatted_data.append({
            "prompt": prompt,
            "relation": item['relation']
        })
    return formatted_data

def Qwen_format(data):
    """
    Converts the data to Qwen format.
    
    Args:
        data (list): List of data points to convert.
    
    Returns:
        list: Data in Qwen format.
    """
    system = "You are a Relation Extraction models at sentence level."
    formatted_data = []
    relation_types = set(item['relation'] for item in data)
    relation_types = sorted(relation_types)  # Sort for consistency
    relations = ", ".join(relation_types)
    for item in data:
        head_marker = f"<e1>{item['head']}</e1>"
        tail_marker = f"<e2>{item['tail']}</e2>"
        item['sentence'] = item['sentence'].replace(item['head'], head_marker).replace(item['tail'], tail_marker)
        query= f"What is the relation type between the two entities in the sentence? {item['sentence']}. Possible relations: {relations}. Asnwer:"
        prompt = f"""<|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {query} <|im_end|>
            <|im_start|>assistant
            """
        formatted_data.append({
            "prompt": prompt,
            "response": item['relation']
        })
    return formatted_data

def prepare_data(data, format_type):
    """
    Prepares the data in the specified format.
    
    Args:
        data (list): List of data points to prepare.
        format_type (str): The format type to convert the data to.
    
    Returns:
        list: Data in the specified format.
    """
    if format_type == 'llama':
        return llama_format(data)
    elif format_type == 'mistral':
        return mistral_format(data)
    elif format_type == 'flan-t5':
        return flan_t5_format(data)
    elif format_type == 'qwen':
        return Qwen_format(data)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data in different formats for finance dataset")
    parser.add_argument("--path_to_data", type=str, required=True, help="Path to the input data JSON file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output files")
    parser.add_argument("--format_type", type=str, choices=['llama', 'mistral', 'flan-t5', 'qwen'], required=True, help="Format type to convert the data to")
    
    args = parser.parse_args()
    
    data = read_json(args.path_to_data)
    formatted_data = prepare_data(data, args.format_type)
    
    output_path = args.output_folder
    write_json(formatted_data, output_path)
    
    print(f"Data has been prepared and saved to {output_path}")