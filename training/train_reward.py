"""
Reward Model Training Script
Trains a reward model on human preference comparisons
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
import wandb
import logging
from tqdm import tqdm

from data.dataset import PreferenceDataset, load_data_from_json, create_preference_pairs
from models.reward_model import RewardModel, RewardModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_reward_model(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    config: dict,
    use_wandb: bool = True
):
    """
    Train reward model on preference data
    
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
    
    # Create preference pairs
    logger.info("Creating preference pairs...")
    train_pairs = create_preference_pairs(train_data)
    val_pairs = create_preference_pairs(val_data)
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = PreferenceDataset(
        data=train_pairs,
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = PreferenceDataset(
        data=val_pairs,
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['reward_model']['comparison_batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['reward_model']['comparison_batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize reward model
    logger.info(f"Initializing reward model: {model_name}")
    reward_model = RewardModel(model_name=model_name, device=str(device))
    
    # Initialize trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        device=str(device)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=config['reward_model']['learning_rate'],
        weight_decay=0.01
    )
    
    # Prepare with accelerator
    reward_model, optimizer, train_loader, val_loader = accelerator.prepare(
        reward_model, optimizer, train_loader, val_loader
    )
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            name="reward-model-training",
            config=config
        )
    
    # Training loop
    num_epochs = config['reward_model']['num_epochs']
    best_val_loss = float('inf')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        reward_model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            metrics = trainer.train_step(batch, optimizer)
            
            train_loss += metrics['loss']
            train_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/loss": metrics['loss'],
                    "train/accuracy": metrics['accuracy'],
                    "train/chosen_reward": metrics['chosen_reward_mean'],
                    "train/rejected_reward": metrics['rejected_reward_mean']
                })
            
            progress_bar.set_postfix({
                "loss": metrics['loss'],
                "acc": metrics['accuracy']
            })
        
        avg_train_loss = train_loss / num_batches
        avg_train_accuracy = train_accuracy / num_batches
        
        # Validation
        reward_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                chosen_rewards = reward_model(
                    batch["input_ids_chosen"],
                    batch["attention_mask_chosen"]
                )
                
                rejected_rewards = reward_model(
                    batch["input_ids_rejected"],
                    batch["attention_mask_rejected"]
                )
                
                # Compute loss
                loss = trainer.compute_loss(chosen_rewards, rejected_rewards)
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                
                val_loss += loss.item()
                val_accuracy += accuracy
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        avg_val_accuracy = val_accuracy / num_val_batches
        
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_accuracy": avg_train_accuracy,
                "val/loss": avg_val_loss,
                "val/accuracy": avg_val_accuracy
            })
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            reward_model.save(checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save epoch checkpoint
        epoch_checkpoint = os.path.join(output_dir, f"epoch_{epoch + 1}.pt")
        reward_model.save(epoch_checkpoint)
    
    logger.info("Training complete!")
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--train-data", type=str, default="./data/splits/train.json", help="Training data path")
    parser.add_argument("--val-data", type=str, default="./data/splits/val.json", help="Validation data path")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/reward_model", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_reward_model(
        model_name=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        config=config,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()

