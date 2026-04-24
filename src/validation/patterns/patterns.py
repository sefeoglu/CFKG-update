
import os
import json
import sys

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    file = "/Users/sefika/phd_projects/CRE_PTM copy/data/fewrel/final/train_wiki.json"
    data = read_json(file)
    relation_set = set()
    for k in data:
        print(k['relation_PID'])
        relation_set.add(k['relation_PID'])
    relation_set = list(relation_set)
    write_json(relation_set, "./test_output.json")