"""
Supervised Fine-Tuning Training Script
Trains a language model on Reddit post-summary pairs
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
import wandb
import logging

from data.dataset import SummarizationDataset, load_data_from_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_sft(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    config: dict,
    use_wandb: bool = True
):
    """
    Train supervised fine-tuning model
    
    Args:
        model_name: Base model name
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON
        output_dir: Output directory for checkpoints
        config: Configuration dictionary
        use_wandb: Whether to use wandb logging
    """
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    logger.info("Loading training data...")
    train_data = load_data_from_json(train_data_path)
    val_data = load_data_from_json(val_data_path)
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = SummarizationDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_input_length=config['model']['max_length'],
        max_output_length=config['model']['summary_max_length']
    )
    
    val_dataset = SummarizationDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_input_length=config['model']['max_length'],
        max_output_length=config['model']['summary_max_length']
    )
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['sft']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['sft']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=config['logging']['log_dir'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=config['training']['fp16'],
        report_to="wandb" if use_wandb else "none",
        run_name="sft-reddit-summarizer"
    )
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            name="sft-training",
            config=config
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    if use_wandb:
        wandb.finish()
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train SFT model")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--train-data", type=str, default="./data/splits/train.json", help="Training data path")
    parser.add_argument("--val-data", type=str, default="./data/splits/val.json", help="Validation data path")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sft", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_sft(
        model_name=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        config=config,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()

