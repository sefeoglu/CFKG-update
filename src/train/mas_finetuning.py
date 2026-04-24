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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_json(path: str) -> Dict:
    """
    Read a JSON file from the given path.
    
    Args:
        path: File path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data: Dict, path: str) -> None:
    """
    Write a JSON file to the given path.
    
    Args:
        data: Dictionary to save
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def tensor_to_serializable(obj: Any) -> Any:
    """
    Convert tensor objects to serializable format for JSON.
    
    Args:
        obj: Object to convert (tensor, dict, list, or other)
        
    Returns:
        Serializable version of input object
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item() if isinstance(obj.item(), (int, float)) else float(obj.item())
        else:
            return [float(x) for x in obj.view(-1).tolist()]
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_to_serializable(v) for v in obj]
    return obj


def freeze_old_params(model) -> Dict[str, torch.Tensor]:
    """
    Freeze model parameters for MAS regularization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of frozen parameter tensors
    """
    return {name: param.detach().clone() for name, param in model.named_parameters()}


# ============================================================================
# MAS TRAINER
# ============================================================================

class MASTrainer(Seq2SeqTrainer):
    """
    Memory-Aware Synapsis Trainer for continual learning.
    
    Extends Seq2SeqTrainer with MAS regularization to prevent catastrophic
    forgetting during continual learning across multiple tasks.
    """
    
    def __init__(self, *args, old_params: Optional[Dict] = None, 
                 importance: Optional[Dict] = None, mas_lambda: float = 1.0, **kwargs):
        """
        Initialize MAS Trainer.
        
        Args:
            old_params: Frozen parameters from previous task
            importance: Parameter importance scores
            mas_lambda: Weight for MAS regularization term
        """
        super().__init__(*args, **kwargs)
        self.old_params = old_params
        self.importance = importance
        self.mas_lambda = mas_lambda

    def compute_loss(self, model, inputs, return_outputs: bool = False, 
                    num_items_in_batch: Optional[int] = None):
        """
        Compute loss with MAS regularization.
        
        Args:
            model: Model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply MAS regularization if parameters and importance are available
        if self.old_params and self.importance:
            mas_reg = 0.0
            for name, param in model.named_parameters():
                if name in self.old_params:
                    mas_reg += (self.importance[name] * (param - self.old_params[name]).pow(2)).sum()
            loss += self.mas_lambda * mas_reg

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# IMPORTANCE COMPUTATION
# ============================================================================

def compute_mas_importance(model, dataloader, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute parameter importance using Fisher information.
    
    Uses Fisher information matrix diagonal to estimate importance of each parameter
    for the current task, which helps prevent catastrophic forgetting.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for computing importance
        device: Device to run computation on
        
    Returns:
        Dictionary mapping parameter names to importance scores
    """
    importance = {name: torch.zeros_like(param) 
                  for name, param in model.named_parameters()}
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            model.zero_grad()
            inputs = {k: v.to(next(model.parameters()).device) 
                     for k, v in batch.items()}

            for param in model.parameters():
                param.requires_grad = True

            with torch.enable_grad():
                outputs = model(**inputs)
                loss = outputs.loss.norm(2)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        importance[name] += param.grad.abs().detach()

    # Normalize importance scores
    num_batches = max(1, len(dataloader))
    for name in importance:
        importance[name] /= num_batches

    return importance


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def set_dataset(dataset_id: str):
    """
    Load dataset from Hugging Face Hub.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'wnut_17')
        
    Returns:
        Loaded dataset
    """
    dataset = load_dataset(dataset_id)
    logger.info(f"Validation set size: {len(dataset['validation'])}")
    return dataset


def set_tokenizer(model_id: str = "google-t5/t5-base"):
    """
    Load tokenizer from Hugging Face Hub.
    
    Args:
        model_id: Model identifier for tokenizer
        
    Returns:
        AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str = "google-t5/t5-base", local: bool = False):
    """
    Load model from Hugging Face Hub or locally.
    
    Args:
        model_id: Model identifier
        local: Load model from local files only
        
    Returns:
        AutoModelForSeq2SeqLM instance
    """
    kwargs = {"device_map": "auto"}
    if local:
        kwargs["local_files_only"] = True
    return AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)


def preprocess_function(sample: Dict, tokenizer, max_source_length: int, 
                       max_target_length: int, padding: str = "max_length") -> Dict:
    """
    Preprocess text samples for T5 model.
    
    Tokenizes input prompts and target relations, preparing them for training.
    
    Args:
        sample: Dictionary with 'prompt' and 'relation' keys
        tokenizer: Tokenizer instance
        max_source_length: Maximum length for input sequences
        max_target_length: Maximum length for target sequences
        padding: Padding strategy ('max_length' or 'longest')
        
    Returns:
        Dictionary with tokenized inputs, attention masks, and labels
    """
    inputs = [item for item in sample["prompt"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_source_length, 
        padding=padding, 
        truncation=True
    )

    # Tokenize targets
    labels = tokenizer(
        text_target=sample["relation"],
        max_length=max_target_length,
        padding=padding,
        truncation=True
    )

    # Replace pad tokens in labels with -100 for loss computation
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(labels: list, preds: list) -> Tuple[list, list]:
    """
    Post-process predictions and labels.
    
    Removes newlines and extracts answer portion after 'Answer:' marker.
    
    Args:
        labels: List of reference labels
        preds: List of predictions
        
    Returns:
        Tuple of (cleaned_preds, cleaned_labels)
    """
    preds = [pred.replace('\n', '').split('Answer:')[-1].strip() for pred in preds]
    labels = [label.replace('\n', '').split('Answer:')[-1].strip() for label in labels]
    return preds, labels


# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def compute_metrics(eval_preds, tokenizer, tasks_path: str) -> Dict[str, float]:
    """
    Compute ROUGE metrics for evaluation.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        tokenizer: Tokenizer instance
        tasks_path: Path to tasks JSON file
        
    Returns:
        Dictionary with ROUGE score metrics
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Handle -100 tokens (padding)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process
    preds_clean, labels_clean = postprocess_text(decoded_labels, decoded_preds)

    # Prepare for ROUGE computation
    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def preprocess_logits_for_metrics(logits, labels) -> torch.Tensor:
    """
    Preprocess logits for metric computation.
    
    Args:
        logits: Model logits
        labels: Target labels
        
    Returns:
        Argmax of logits
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(model_id: str, data_path: str, tasks_path: str, task_id: str,
                batch_size_train: int = 8, batch_size_eval: int = 8, 
                epochs: int = 10, local: bool = False, use_mas: bool = False, 
                mas_lambda: float = 1.0, old_params: Optional[Dict] = None, 
                importance: Optional[Dict] = None) -> Tuple:
    """
    Train T5 model with optional MAS regularization.
    
    Implements full training pipeline including data loading, preprocessing,
    model configuration, and training with optional continual learning support.
    
    Args:
        model_id: Model identifier from Hugging Face Hub
        data_path: Path to dataset
        tasks_path: Path to tasks JSON file
        task_id: Task identifier
        batch_size_train: Training batch size
        batch_size_eval: Evaluation batch size
        epochs: Number of training epochs
        local: Load model from local files
        use_mas: Use MAS regularization for continual learning
        mas_lambda: Weight for MAS regularization
        old_params: Frozen parameters from previous task
        importance: Parameter importance scores from previous task
        
    Returns:
        Tuple of (model, tokenizer, trainer, importance_scores)
    """
    logger.info(f"Starting training for task: {task_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data and model
    dataset = set_dataset(data_path)
    tokenizer = set_tokenizer(model_id)
    base_model = load_model(model_id, local)
    logger.info(f"Loaded model: {model_id}")

    # Configure LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["k", "q", "v", "o"]
    )
    
    model = get_peft_model(base_model, lora_config)
    logger.info("Applied LoRA configuration")

    # Compute max lengths from dataset
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["validation"]]).map(
        lambda x: tokenizer(x["prompt"], truncation=True),
        batched=True,
        remove_columns=["prompt", "relation"]
    )
    max_source_length = max(len(x) for x in tokenized_inputs["input_ids"])

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"]]).map(
        lambda x: tokenizer(x["relation"], truncation=True),
        batched=True,
        remove_columns=["prompt", "relation"]
    )
    max_target_length = max(len(x) for x in tokenized_targets["input_ids"])

    logger.info(f"Max source length: {max_source_length}")
    logger.info(f"Max target length: {max_target_length}")

    # Preprocess dataset
    preprocess_fn = lambda x: preprocess_function(
        x, tokenizer, max_source_length, max_target_length
    )
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "relation"]
    )
    logger.info(f"Dataset preprocessing complete. Features: {list(tokenized_dataset['train'].features)}")

    # Data collator for sequence-to-sequence models
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # Training data loader for importance computation
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size_train,
        collate_fn=data_collator
    )

    # Setup training arguments
    repository_id = f"{model_id}-{task_id}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        predict_with_generate=True,
        fp16=False,  # Avoid overflow issues
        learning_rate=1e-3,
        num_train_epochs=epochs,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        push_to_hub=False,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 1},
        remove_unused_columns=False,
        weight_decay=0.1
    )

    # Create trainer (MAS or standard)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "compute_metrics": lambda x: compute_metrics(x, tokenizer, tasks_path),
    }
    
    if use_mas and old_params is not None:
        trainer_kwargs.update({
            "old_params": old_params,
            "importance": importance,
            "mas_lambda": mas_lambda
        })
        trainer = MASTrainer(**trainer_kwargs)
        logger.info("Using MASTrainer with regularization")
    else:
        trainer = Seq2SeqTrainer(**trainer_kwargs)
        logger.info("Using standard Seq2SeqTrainer")

    # Train model
    logger.info("Starting training...")
    trainer.train()
    trainer.save_state()
    trainer.save_model()
    logger.info(f"Training complete for task: {task_id}")

    # Compute importance for next task
    new_importance = compute_mas_importance(model, train_dataloader, device)
    merged_model = model.merge_and_unload()

    return merged_model, tokenizer, trainer, new_importance


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
