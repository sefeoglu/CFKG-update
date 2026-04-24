# pip install evaluate bert-score
import evaluate
import json
import os
import sys
import argparse 

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

