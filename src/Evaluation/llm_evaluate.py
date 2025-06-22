"""Generates responses for prompts using a language model defined in Hugging Face."""
"""Created by: Sefika"""
import sys
import os

import json
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from peft import  get_peft_model, LoraConfig, TaskType
class LLM(object):

    def __init__(self, model_id="google/flan-t5-xl"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-xl".
        """

        
        self.model, self.tokenizer = self.get_model(model_id)



    def get_model(self, model_id="google/flan-t5-base"):
        """_summary_

        Args:
            model_id (str, optional): LLM name at HuggingFace . Defaults to "google/flan-t5-xl".

        Returns:
            model: model from Hugging Face
            tokenizer: tokenizer of this model
        """
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        model = T5ForConditionalGeneration.from_pretrained(model_id,
                                                    device_map="auto",
                                                    load_in_8bit=False,
                                                    torch_dtype=torch.float16
                                                    )
        return model,tokenizer



    def get_model_decoder(self, model_id="meta-llama/Llama-2-7b-chat-hf"):
        """loades the model from Hugging Face such llama and mistral

        Args:
            model_id (str, optional): _description_. Defaults to "meta-llama/Llama-2-7b-chat-hf".

        Returns:
            model: loaded model
            tokenizer: loaded tokenizer
        """

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                    device_map="auto",
                                                    load_in_8bit=False,
                                                    torch_dtype=torch.float16,
                                                    # max_memory=self.maxmem
                                                    )
        # max_memory=self.maxmem
        return model,tokenizer

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_prediction(model,tokenizer, prompt, length=250,stype='greedy'):

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096, truncation=True, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs,  pad_token_id=tokenizer.eos_token_id, max_new_tokens=length)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return response



def evaluate_model(experiment_id, task_id, model, tokenizer, out_folder, current_task=True):

    if current_task:
      input_path = "/home/sefie08/projects/CRE-test/CRE-mistral/CRE-evaluation/llama_format_data/test/run_{0}/task{1}/test_1.json".format(experiment_id, task_id)
      data = read_json(input_path)
      out_pred_path = "/home/sefie08/projects/CRE-test/CRE-mistral/CRE-evaluation/llama-results/fewrel/llama/m_10/KMmeans_CRE_tacred_{0}/task_{1}_current_task_pred.json".format(experiment_id, task_id)
      out_acc_path = "/home/sefie08/projects/CRE-test/CRE-mistral/CRE-evaluation/llama-results/fewrel/llama/m_10/KMmeans_CRE_tacred_{0}/task_{1}_current_task_result.json".format(experiment_id, task_id)
    else:
      data = []
      for t in range(1, task_id+1):
          input_path = "/scratch/sefie08/projects/llm_forget/finance/run_{0}/task_{1}/test.json".format(experiment_id, t)
          task_data = read_json(input_path)
          data.extend(task_data)
      out_pred_path = f"{out_folder}/model_{0}/task_{1}_seen_task.json".format(experiment_id, task_id)
      out_acc_path = f"{out_folder}/model_{0}/task_{1}_seen_task_result.json".format(experiment_id, task_id)
    responses = []
    relations = []

    for j, item in enumerate(data):

      prompt = item['prompt']
      relations.append(item['relation'])


      response = get_prediction(model, tokenizer, prompt)

      print('test:', j)

      if len(response) == 0:
          print("No response")
          responses.append("")
      else:
          response = response[0]
          responses.append({"predict":response})
    acc = accuracy_score(relations, [r['predict'] for r in responses])
    write_json({"accuracy": acc}, out_acc_path)

    write_json(responses, out_pred_path)

def main(outfolder):
    """Main function to evaluate the model on multiple tasks and experiments."""
    out_folder = outfolder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Loop through experiments and tasks

    for experiment_id in range(1, 6):

        for task_id in range(1, 6):

            model_id = f"Sefika/base_fs_finance_random_{experiment_id}_{task_id}"

            llm_instance = LLM(model_id)
            tokenizer = llm_instance.tokenizer
            model = llm_instance.model
            evaluate_model(experiment_id, task_id, model, tokenizer, out_folder, False)

if __name__ =="__main__":
    if len(sys.argv) != 2:
        print("Usage: python llm_evaluate.py <output_folder>")
        sys.exit(1)
    output_folder = sys.argv[1]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Output folder: {output_folder}")
    print("Starting evaluation...")

    main(output_folder)