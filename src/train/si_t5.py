"""
Memory-Aware Synapsis (MAS) Fine-tuning

Continuous instruction fine-tuning of FLAN-T5 model for Relation Extraction
with Memory-Aware Synapsis regularization for continual learning.

Author: [Your Name]
Date: 2024
"""

import sys
import os
import json
import torch
import logging
import numpy as np
from train.mas_finetuning import freeze_old_params
import wandb
import evaluate
import nltk
from datetime import datetime
from random import randrange
from typing import Dict, Tuple, Optional, Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig,
    DataCollatorForSeq2Seq, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import HfFolder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt")

# Load metric
metric = evaluate.load("rouge")


# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_EXPERIMENTS = 5
NUM_TASKS = 5
EPOCHS = 10
BATCH_SIZE = 8
MODEL_ID = "google/flan-t5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path):
    """Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    """Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def postprocess_text(labels, preds):
    """Standard post-processing for relation extraction outputs."""
    preds = [pred.replace('\n','').split('Answer:')[-1].strip() for pred in preds]
    labels = [label.replace('\n','').split('Answer:')[-1].strip() for label in labels]
    return preds, labels

# --- Synaptic Intelligence (SI) Implementation ---

class SISeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Trainer implementing Synaptic Intelligence (SI) regularizer.
    It tracks parameter importance to preserve knowledge of previous tasks.
    """
    def __init__(self, *args, si_lambda=1.0, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_lambda = si_lambda
        self.epsilon = epsilon
        self.prev_params = {}
        self.omega = {}
        self.w = {}

        # Initialize parameter tracking for SI
        for n, p in self.model.named_parameters():
            self.prev_params[n] = p.clone().detach()
            self.omega[n] = torch.zeros_like(p)
            self.w[n] = torch.zeros_like(p)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard cross-entropy loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Add SI penalty to mitigate forgetting
        si_penalty = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                delta = p - self.prev_params[n]
                si_penalty += (self.omega[n] * delta ** 2).sum()
        
        loss += self.si_lambda * si_penalty
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        # Accumulate importance (path integral) after backward pass
        loss = super().training_step(*args, **kwargs)
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.w[n] -= p.grad.detach() * (p.detach() - self.prev_params[n])
        return loss

    def update_omega(self):
        """Update importance weights (omega) after completing a task."""
        for n, p in self.model.named_parameters():
            delta = p.detach() - self.prev_params[n]
            # Small epsilon to avoid division by zero
            self.omega[n] += self.w[n] / (delta ** 2 + self.epsilon)
            self.prev_params[n] = p.clone().detach()
            self.w[n].zero_()

# --- Training Orchestrator ---

def set_tokenizer(model_id="google/flan-t5-base"):
    return AutoTokenizer.from_pretrained(model_id)

def load_model(model_id="google/flan-t5-base", local=False):
    if local:
        return AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", local_files_only=True)
    return AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process for cleaner comparison
    grounds, cleaned_preds = postprocess_text(decoded_labels, decoded_preds)
    
    # Compute ROUGE scores
    result = metric.compute(
        predictions=["\n".join(p.strip()) for p in decoded_preds],
        references=["\n".join(l.strip()) for l in decoded_labels],
        use_stemmer=True
    )
    return {k: round(v * 100, 4) for k, v in result.items()}

def train_model(model_id, data_path, tasks_path, task_id, bs_train, bs_eval, epochs, lora_config, local=False):
    tokenizer = set_tokenizer(model_id)
    dataset = load_dataset(data_path)
    base_model = load_model(model_id, local)
    model = get_peft_model(base_model, lora_config)

    # Tokenization preprocessing
    def preprocess_fn(sample):
        model_inputs = tokenizer(sample["prompt"], max_length=512, padding="max_length", truncation=True)
        labels = tokenizer(text_target=sample["relation"], max_length=64, padding="max_length", truncation=True)
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_fn, batched=True, remove_columns=["prompt", "relation"])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    repository_id = f"{model_id.split('/')[-1]}-{task_id}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,
        predict_with_generate=True,
        learning_rate=1e-3,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        weight_decay=0.1
    )

    trainer = SISeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        si_lambda=1.0
    )

    trainer.train()
    trainer.update_omega() # Important: Update SI weights after task completion
    trainer.save_model()
    
    return model.merge_and_unload(), tokenizer, trainer

def get_prediction(model, tokenizer, prompt, length=250):
    """Utility to generate prediction from a trained model."""
    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(inputs, max_new_tokens=length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)



# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_experiments() -> str:
    """
    Run complete experiment pipeline with multiple tasks.
    
    Executes continual learning experiments where:
    - Task 1: Train base model without regularization
    - Tasks 2-5: Train with MAS regularization to prevent forgetting
    
    Returns:
        String with training logs and timing information
    """
    logs = ""

    for experiment_id in range(1, NUM_EXPERIMENTS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {experiment_id}")
        logger.info('='*60)
        
        wandb.init(
            project="corona_mas_fscre_cl",
            name=f"experiment-{experiment_id}-1",
            config={"epochs": EPOCHS, "batch_size": BATCH_SIZE}
        )

        # Task 1: Base model training (no regularization)
        dataset_path = f'/content/corona/train/run_{experiment_id}/task_1'
        tasks_path = f"/content/corona/tasks/run_{experiment_id}/task_1.json"
        
        logger.info(f"Training base model for experiment {experiment_id}...")
        start_time = datetime.now()
        model, tokenizer, trainer, importance = train_model(
            MODEL_ID, dataset_path, tasks_path, "task_1",
            batch_size_train=BATCH_SIZE, batch_size_eval=BATCH_SIZE, 
            epochs=EPOCHS, local=False
        )
        end_time = datetime.now()
        
        train_time = f"Base Train. Experiment {experiment_id}. Task 1. Duration: {end_time - start_time}\n"
        logs += train_time
        logger.info(train_time)
        
        # Save base model
        model.save_pretrained(f"tacred_mas_cl_fscre_{experiment_id}/FSCRE_5_1/", from_pt=True)
        model.push_to_hub(f"tacred_mas_cl_fscre_{experiment_id}_1", private=True)
        logger.info(f"Saved base model for experiment {experiment_id}")
        
        wandb.finish()

        # Tasks 2-5: Continual learning with MAS regularization
        for task_num in range(2, NUM_TASKS + 1):
            wandb.init(
                project="corona_mas_fscre_cl",
                name=f"experiment-{experiment_id}-{task_num}",
                config={"epochs": EPOCHS, "batch_size": BATCH_SIZE}
            )

            dataset_path = f'/content/corona/train/run_{experiment_id}/task_{task_num}'
            tasks_path = f"/content/corona/tasks/run_{experiment_id}/task_{task_num}.json"
            base_model_id = f"Sefika/tacred_mas_cl_fscre_{experiment_id}_{task_num-1}"

            logger.info(f"\nTraining task {task_num} with MAS regularization...")
            
            start_time = datetime.now()
            model, tokenizer, trainer, importance = train_model(
                base_model_id, dataset_path, tasks_path, f"task_{task_num}",
                batch_size_train=BATCH_SIZE, batch_size_eval=BATCH_SIZE,
                epochs=EPOCHS, local=False, use_mas=True, mas_lambda=1.0,
                old_params=freeze_old_params(trainer.model),
                importance=importance
            )
            end_time = datetime.now()

            train_time = f"MAS Train. Experiment {experiment_id}. Task {task_num}. Duration: {end_time - start_time}\n"
            logs += train_time
            logger.info(train_time)

            # Save model
            model.save_pretrained(
                f"tacred_mas_cl_fscre_{experiment_id}/FSCRE_5_{task_num}/", 
                from_pt=True
            )
            model.push_to_hub(f"tacred_mas_cl_fscre_{experiment_id}_{task_num}", private=True)
            logger.info(f"Saved model for experiment {experiment_id}, task {task_num}")
            
            wandb.finish()

    logger.info("\n" + "="*60)
    logger.info("All experiments completed!")
    logger.info("="*60)
    
    return logs


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the script.
    
    Before running:
    1. Set environment variables:
       - HF_TOKEN: Hugging Face API token
       - WANDB_API_KEY: Weights & Biases API key
    
    2. Ensure dataset and task files are in correct paths:
       - /content/corona/train/run_*/task_*
       - /content/corona/tasks/run_*/task_*.json
    
    3. Configure NUM_EXPERIMENTS, NUM_TASKS, EPOCHS, BATCH_SIZE as needed
    """
    try:
        logs = run_experiments()
        print("\n" + logs)
    except Exception as e:
        logger.error(f"Error during experiments: {e}", exc_info=True)
        raise
