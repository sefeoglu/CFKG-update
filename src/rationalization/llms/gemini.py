from http import client
from google import genai
import json
import os
import argparse
from tqdm import tqdm
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
def generate_text_with_gemini(question, config):
    """
    Gemini 2.5 Pro modelini kullanarak basit bir metin oluşturma örneği.
    """

    client = genai.Client(api_key=config["gemini_api_key"])

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=question +"Please only answer with score, explanation and suggestions in a short form."
    )
    return response.text


def bulk_test(templates, out_file, config):
    results = []
    for item in tqdm(templates, desc="Processing templates"):
       
        template = item['rationale_prompt']
    

        prediction = generate_text_with_gemini(template, config)


        item['predictions_rationale'] = prediction.split('\n')
     

        results.append(item)
        write_json(results, out_file)
        
        time.sleep(0.10)  # API rate limitine dikkat etmek için kısa bir bekleme ekleyin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine contextual information from multiple sources.")
    parser.add_argument("--input_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/rationale_nli/fewrel_mas_top_prob_5/model_5/test_pred_8_prompts.json", help="Path to the input JSON file containing contextual information.")
    parser.add_argument("--output_file", type=str, default="/Users/sefika/phd_projects/llm-catastrophic-re/results_rationale_nli/gemini/fewrel_mas_top_prob_5/model_5/test_pred_8_prompts_1.json", help="Path to the output JSON file to save combined information.")
    parser.add_argument("--config", type=str, default="/Users/sefika/phd_projects/converse_relations/data/gemini_key.json", help="Path to the config JSON file.")
    
    args = parser.parse_args()
    input_data = read_json(args.input_file)[4324:]
    out_path = args.output_file
    config_path = args.config
    config = read_json(config_path)
    bulk_test(input_data, out_path, config)
    print(f"Predictions saved to {out_path}")
