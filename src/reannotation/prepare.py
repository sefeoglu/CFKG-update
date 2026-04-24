import os
import json
import sys
from qwikidata.sparql import return_sparql_query_results
import time
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def sparql_query(pid, lang="en"):
    return f"""
    SELECT ?altLabel WHERE {{
      wd:{pid} skos:altLabel ?altLabel .
      FILTER(LANG(?altLabel) = "{lang}")
    }}
    ORDER BY ?altLabel
    """

def query_wikidata(sparql_query):
    """
    Query Wikidata SPARQL endpoint and return results.
    """

    try:
        results = return_sparql_query_results(sparql_query)
        print(results)
        return results
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
        return None
    
def get_alt_labels(pid, lang="en"):
    query = sparql_query(pid, lang)
    results = query_wikidata(query)
    alt_labels = []
    if results:
        for result in results["results"]["bindings"]:
            alt_label = result["altLabel"]["value"]
            alt_labels.append(alt_label)
    return alt_labels

def all_data_from_file(file_path, valid_relations_path):
    data = read_json(file_path)
    valid_relations = read_json(valid_relations_path)
    
    count = 0
    canonicals = {}
    print(f"Total relations to process: {len(valid_relations)}")
    for key, value in data.items():
        if key not in valid_relations:
            continue
        pid = key
        label = value[0]
        description = value[1]
        alt_labels = get_alt_labels(pid)
        print(f"PID: {pid}, Alt Labels: {alt_labels}")
        time.sleep(5)  # To avoid overwhelming the SPARQL endpoint
        count += 1
        row = {
            "label": label,
            "description": description,
            "alt_labels": alt_labels
        }
        canonicals[pid] = row
        # if count >= 10:
        #     break
        write_json(canonicals, "./output/canonicals.json")

if __name__ == "__main__":
    file="/Users/sefika/phd_projects/llm-catastrophic-re/dataset/canonical_relation_maps/fewrel/pid2name.json"
    valid_relations ="/Users/sefika/phd_projects/llm-catastrophic-re/src/validation/patterns/test_output.json"
    all_data_from_file(file,valid_relations)
