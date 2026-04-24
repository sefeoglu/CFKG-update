import openai
from http import client
from google import genai
import json
import os
import json
import sys
import argparse
from tqdm import tqdm

import time
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-3])
sys.path.append(PREFIX_PATH)


def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# logging.basicConfig(level=logging.INFO, filename=f"{PREFIX_PATH}/logging/contextual_information.log", format='%(asctime)s - %(levelname)s - %(message)s')


def run_gpt_chat(config):
    # print(config)
    user_query = config['user_query']
   
    API_KEY = config['openai_api_key']
    model = config['model']

    openai.api_key = API_KEY

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": user_query}
        ],
        temperature=0,          # Makes output deterministic
        seed=42                 # Ensures repeatability across runs
    )

    return response['choices'][0]['message']['content'].split('\n')



# def prediction_gpt(input_data, config, out_path):
#     all_predictions = []
#     for idx, item in tqdm(enumerate(input_data), total=len(input_data), desc="Processing items with GPT"):
#         template_1 = item['template_1']
#         template_2 = item['template_2']
#         item['predictions_1'] = run_gpt_chat({'user_query': template_1, 'openai_api_key': config['openai_api_key'], 'model': config['model']})
#         item['predictions_2'] = run_gpt_chat({'user_query': template_2, 'openai_api_key': config['openai_api_key'], 'model': config['model']})
#         all_predictions.append(item)

#         with open(out_path, 'w') as f:
#             json.dump(all_predictions, f, indent=4)

#     return all_predictions
def bulk_test(templates, out_file, config):
    results = []
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for item in tqdm(templates, desc="Processing templates"):
       
        template = item['rationale_prompt']
    

        
        configuration = {'user_query': template, 'openai_api_key': config['openai_api_key'], 'model': config['model']}
        item['prediction'] =  run_gpt_chat(configuration) 
        
        results.append(item)
        write_json(results, out_file)
        
        time.sleep(0.10)  # API rate limitine dikkat etmek için kısa bir bekleme ekleyin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine contextual information from multiple sources.")
    parser.add_argument("--input_file", type=str, default="../llm-catastrophic-re/results_all/rationale_nli/fewrel_mas_top_prob_5/model_1/test_pred_8_prompts.json", help="Path to the input JSON file containing contextual information.")
    parser.add_argument("--output_file", type=str, default="../llm-catastrophic-re/results_all/results_rationale_nli/gpt_nli/fewrel_mas_top_prob_5/model_1/test_pred_8_prompts_1.json", help="Path to the output JSON file to save combined information.")
    parser.add_argument("--config", type=str, default="../converse_relations/data/gpt_key.json", help="Path to the config JSON file.")
    
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    input_data = json.load(open(args.input_file, 'r'))
    out_path = args.output_file
    predictions = bulk_test(input_data, out_path, config)
    print(f"Predictions saved to {out_path}")