"""
Elastic Weight Consolidation (EWC) Fine-tuning

Continuous instruction fine-tuning of FLAN-T5 model for Relation Extraction
with Elastic Weight Consolidation regularization for continual learning.

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
EWC_LAMBDA = 100  # Weight for EWC regularization
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


def freeze_old_params(model) -> Dict[str, torch.Tensor]:
    """
    Freeze model parameters for EWC regularization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of frozen parameter tensors
    """
    return {name: param.detach().clone() for name, param in model.named_parameters()}


# ============================================================================
# ELASTIC WEIGHT CONSOLIDATION TRAINER
# ============================================================================

class EWCTrainer(Seq2SeqTrainer):
    """
    Elastic Weight Consolidation Trainer for continual learning.
    
    Extends Seq2SeqTrainer with EWC regularization to prevent catastrophic
    forgetting during continual learning across multiple tasks.
    
    EWC protects important parameters by penalizing changes proportional to
    the Fisher Information matrix computed from previous tasks.
    """
    
    def __init__(self, *args, fisher: Optional[Dict] = None, 
                 old_params: Optional[Dict] = None, ewc_lambda: float = 0.4, **kwargs):
        """
        Initialize EWC Trainer.
        
        Args:
            fisher: Fisher Information matrix from previous task
            old_params: Frozen parameters from previous task
            ewc_lambda: Weight for EWC regularization term
        """
        super().__init__(*args, **kwargs)
        self.fisher = fisher
        self.old_params = old_params
        self.ewc_lambda = ewc_lambda

    def compute_loss(self, model, inputs, return_outputs: bool = False, 
                    num_items_in_batch: Optional[int] = None):
        """
        Compute loss with EWC regularization.
        
        EWC loss = standard_loss + ewc_lambda * sum(fisher * (param - old_param)^2)
        
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

        # Add EWC loss if fisher and old_params are provided
        if self.old_params is not None and self.fisher is not None:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in self.fisher:
                    fisher_val = self.fisher[name]
                    old_param = self.old_params[name]
                    # EWC penalty: fisher * (param - old_param)^2
                    ewc_loss += (fisher_val * (param - old_param).pow(2)).sum()
            loss = loss + self.ewc_lambda * ewc_loss

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# FISHER INFORMATION COMPUTATION
# ============================================================================

def compute_fisher(model, dataloader) -> Dict[str, torch.Tensor]:
    """
    Compute the Fisher Information matrix for a given model and dataloader.
    
    The Fisher Information matrix is the expected value of the squared gradients.
    It identifies important parameters that should be protected during continual learning.
    
    Formula: F = E[(∂L/∂θ)²]
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for computing Fisher Information
        
    Returns:
        Dictionary mapping parameter names to Fisher Information values
    """
    fisher = {
        name: torch.zeros_like(param, device=param.device)
        for name, param in model.named_parameters() if param.requires_grad
    }
    
    model.eval()
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            model.zero_grad()
            inputs = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

            # Enable gradients for Fisher computation
            for param in model.parameters():
                param.requires_grad = True

            with torch.enable_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()

                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad and name in fisher:
                        fisher[name] += param.grad.detach() ** 2

            total_samples += 1

    # Average Fisher values over all samples
    for name in fisher:
        fisher[name] /= max(1, total_samples)

    logger.info(f"Computed Fisher Information for {len(fisher)} parameters")
    return fisher


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


def set_tokenizer(model_id: str = "google/flan-t5-base"):
    """
    Load tokenizer from Hugging Face Hub.
    
    Args:
        model_id: Model identifier for tokenizer
        
    Returns:
        AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str = "google/flan-t5-base", local: bool = False):
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
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_source_length, 
        padding=padding, 
        truncation=True
    )

    labels = tokenizer(
        text_target=sample["relation"],
        max_length=max_target_length,
        padding=padding,
        truncation=True
    )

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

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds_clean, labels_clean = postprocess_text(decoded_labels, decoded_preds)

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

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
                epochs: int = 10, local: bool = False, apply_ewc: bool = False, 
                fisher: Optional[Dict] = None) -> Tuple:
    """
    Train T5 model with optional EWC regularization.
    
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
        apply_ewc: Use EWC regularization for continual learning
        fisher: Fisher Information matrix from previous task
        
    Returns:
        Tuple of (model, tokenizer, trainer, fisher_info)
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

    # Setup training arguments
    repository_id = f"{model_id}-{task_id}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        predict_with_generate=True,
        fp16=False,
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

    # Prepare old parameters and fisher info for EWC
    if apply_ewc:
        old_params = freeze_old_params(model)
    else:
        old_params = None
        fisher = None

    # Create trainer
    trainer = EWCTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=lambda x: compute_metrics(x, tokenizer, tasks_path),
        fisher=fisher,
        old_params=old_params,
        ewc_lambda=EWC_LAMBDA
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    trainer.save_state()
    trainer.save_model()
    trainer.model.save_pretrained(repository_id)
    logger.info(f"Training complete for task: {task_id}")

    # Compute Fisher Information for next task
    merged_model = model.merge_and_unload()
    new_fisher = compute_fisher(merged_model, trainer.get_train_dataloader())

    return merged_model, tokenizer, trainer, new_fisher


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_experiments() -> str:
    """
    Run complete experiment pipeline with multiple tasks.
    
    Executes continual learning experiments where:
    - Task 1: Train base model without EWC
    - Tasks 2-5: Train with EWC regularization to prevent forgetting
    
    Returns:
        String with training logs and timing information
    """
    logs = ""

    for experiment_id in range(1, NUM_EXPERIMENTS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {experiment_id}")
        logger.info('='*60)
        
        wandb.init(
            project="corona-ewc-100-correct",
            name=f"experiment-{experiment_id}-1",
            config={"epochs": EPOCHS, "batch_size": BATCH_SIZE}
        )

        # Task 1: Base model training (no EWC)
        dataset_path = f'/content/corona/train/run_{experiment_id}/task_1'
        tasks_path = f"/content/corona/tasks/run_{experiment_id}/task_1.json"
        
        logger.info(f"Training base model for experiment {experiment_id}...")
        start_time = datetime.now()
        model, tokenizer, trainer, fisher = train_model(
            MODEL_ID, dataset_path, tasks_path, "task_1",
            batch_size_train=BATCH_SIZE, batch_size_eval=BATCH_SIZE, 
            epochs=EPOCHS, local=False, apply_ewc=False
        )
        end_time = datetime.now()
        
        train_time = f"Base Train. Experiment {experiment_id}. Task 1. Duration: {end_time - start_time}\n"
        logs += train_time
        logger.info(train_time)
        
        # Save base model
        model.save_pretrained(f"ewc_corona_{experiment_id}/task_1/", from_pt=True)
        model.push_to_hub(f"ewc_fscre_corona_{experiment_id}_1", private=True)
        logger.info(f"Saved base model for experiment {experiment_id}")
        
        wandb.finish()

        # Tasks 2-5: Continual learning with EWC regularization
        for task_num in range(2, NUM_TASKS + 1):
            wandb.init(
                project="corona-ewc-100-correct",
                name=f"experiment-{experiment_id}-{task_num}",
                config={"epochs": EPOCHS, "batch_size": BATCH_SIZE}
            )

            dataset_path = f'/content/corona/train/run_{experiment_id}/task_{task_num}'
            tasks_path = f"/content/corona/tasks/run_{experiment_id}/task_{task_num}.json"
            base_model_id = f"Sefika/ewc_fscre_corona_{experiment_id}_{task_num-1}"

            logger.info(f"\nTraining task {task_num} with EWC regularization...")
            
            start_time = datetime.now()
            model, tokenizer, trainer, fisher = train_model(
                base_model_id, dataset_path, tasks_path, f"task_{task_num}",
                batch_size_train=BATCH_SIZE, batch_size_eval=BATCH_SIZE,
                epochs=EPOCHS, local=False, apply_ewc=True, fisher=fisher
            )
            end_time = datetime.now()

            train_time = f"EWC Train. Experiment {experiment_id}. Task {task_num}. Duration: {end_time - start_time}\n"
            logs += train_time
            logger.info(train_time)

            # Save model
            model.save_pretrained(f"ewc_corona_{experiment_id}/task_{task_num}/", from_pt=True)
            model.push_to_hub(f"ewc_fscre_corona_{experiment_id}_{task_num}", private=True)
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
