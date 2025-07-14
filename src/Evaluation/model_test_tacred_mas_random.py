import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from sklearn.metrics import accuracy_score
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
import configparser

class LLM(object):

    def __init__(self, model_id="google/flan-t5-base"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-base".
        """
        self.model, self.tokenizer = self.get_model(model_id)



    def get_model(self, model_id="google/flan-t5-base"):
        """_summary_

        Args:
            model_id (str, optional): LLM name at HuggingFace . Defaults to "google/flan-t5-base".

        Returns:
            model: model from Hugging Face
            tokenizer: tokenizer of this model
        """
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                    device_map="auto",
                                                    load_in_8bit=False,
                                                    torch_dtype=torch.float16
                                                    )
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

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096,return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs,  pad_token_id=tokenizer.eos_token_id, max_new_tokens=length)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return response



def evaluate_model(base_out_url, base_test_url, experiment_id, task_id, model, tokenizer, current_task=True):

    if current_task:

      input_path = base_test_url + f"/run{experiment_id}/{task_id}/test.json"
      data = read_json(input_path)
      out_pred_path = base_out_url + f"/model_{experiment_id}/task_{task_id}_current_task_pred.json"
      out_acc_path = base_out_url + f"/model_{experiment_id}/task_{task_id}_current_task_result.json"

    else:

      data = []

      for t in range(1, task_id+1):
          input_path = base_test_url + f"/run{experiment_id}/task{t}/test.json"
          task_data = read_json(input_path)
          data.extend(task_data)
          
      out_pred_path = base_out_url + f"/model_{experiment_id}/task_{task_id}_seen_task.json"
      out_acc_path = base_out_url + f"/model_{experiment_id}/task_{task_id}_seen_task_result.json"

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
          #print(response[0])
          responses.append({"predict":response[0]})
          
    y_true = relations

    preds = [line['predict'] for line in responses]
    acc = accuracy_score(y_true, preds)
    result = [{"acc":acc}]

    write_json(responses, out_pred_path)
    write_json(result,out_acc_path)

def main(base_url, base_test_url):

    print("==========Start==========")

    for experiment_id in range(3, 6):

        for task_id in range(1, 9):

            model_id = f"Sefika/mas_fs_tacred_random_{experiment_id}_{task_id}"

            llm_instance = LLM(model_id)
            tokenizer = llm_instance.tokenizer
            model = llm_instance.model

            evaluate_model(base_url,base_test_url, experiment_id, task_id, model, tokenizer, False)

    print("==========End===========")

if __name__ =="__main__":
    base_url =  "/scratch/sefie08/projects/FSCRE/tacred/mas_flan_random"
    base_test_url = "/scratch/sefie08/projects/FSCRE/tacred/5way5shot-test"
    # config = configparser.ConfigParser()
    # config.read("config.ini")
    # base_url = config["DEFAULT"]["base_url"]
    # base_test_url = config["DEFAULT"]["base_test_url"]
    main(base_url, base_test_url)


