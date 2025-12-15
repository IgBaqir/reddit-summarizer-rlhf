"""
PPO Training Script
Trains policy model using Proximal Policy Optimization
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import wandb
import logging
from tqdm import tqdm

from data.dataset import PPODataset, load_data_from_json
from models.ppo_trainer import PPOTrainer
from models.reward_model import RewardModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_ppo(
    model_path: str,
    reward_model_path: str,
    train_data_path: str,
    output_dir: str,
    config: dict,
    use_wandb: bool = True
):
    """
    Train policy model using PPO
    
    Args:
        model_path: Path to SFT model checkpoint
        reward_model_path: Path to reward model checkpoint
        train_data_path: Path to training data JSON
        output_dir: Output directory for checkpoints
        config: Configuration dictionary
        use_wandb: Whether to use wandb logging
    """
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model
    logger.info(f"Loading policy model from {model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(model_path)
    policy_model.to(device)
    
    # Load reward model
    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = RewardModel(model_name=config['model']['base_model'], device=str(device))
    reward_model.load(reward_model_path)
    reward_model.to(device)
    
    # Load data
    logger.info("Loading training data...")
    train_data = load_data_from_json(train_data_path)
    
    # Create dataset
    train_dataset = PPODataset(
        data=train_data,
        tokenizer=tokenizer,
        max_input_length=config['model']['max_length'],
        max_output_length=config['model']['summary_max_length']
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['ppo']['generation_batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=str(device),
        clip_epsilon=config['ppo']['clip_epsilon'],
        gamma=config['ppo']['gamma'],
        lam=config['ppo']['lam'],
        kl_coef=config['ppo']['kl_coef'],
        value_coef=config['ppo']['value_coef'],
        entropy_coef=config['ppo']['entropy_coef']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['ppo']['learning_rate'],
        weight_decay=0.01
    )
    
    # Prepare with accelerator
    policy_model, optimizer, train_loader = accelerator.prepare(
        policy_model, optimizer, train_loader
    )
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            name="ppo-training",
            config=config
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    num_ppo_epochs = config['ppo']['max_ppo_epochs']
    
    logger.info("Starting PPO training...")
    
    for epoch in range(10):  # Outer loop: number of data passes
        logger.info(f"Data Epoch {epoch + 1}")
        
        progress_bar = tqdm(train_loader, desc=f"PPO Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            posts = batch["post"]
            
            # Generate responses
            logger.debug(f"Generating responses for batch {batch_idx}")
            generation_outputs = trainer.generate_responses(
                prompts=posts,
                max_length=config['model']['summary_max_length'],
                temperature=config['model']['temperature'],
                top_p=config['model']['top_p']
            )
            
            generated_texts = generation_outputs["generated_texts"]
            
            # Extract only the generated summaries (remove prompt)
            summaries = []
            for i, (post, gen_text) in enumerate(zip(posts, generated_texts)):
                # Remove the post from generated text
                if gen_text.startswith(post):
                    summary = gen_text[len(post):].strip()
                else:
                    summary = gen_text
                summaries.append(summary)
            
            # Compute rewards
            logger.debug("Computing rewards...")
            rewards = trainer.compute_rewards(posts, summaries)
            
            # Compute old log probs (for PPO)
            # We need to tokenize the full sequences
            full_texts = [f"{p} {tokenizer.eos_token} {s}" for p, s in zip(posts, summaries)]
            full_inputs = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=640
            ).to(device)
            
            old_log_probs = trainer.compute_logprobs(
                full_inputs["input_ids"],
                full_inputs["attention_mask"],
                policy_model
            )
            
            # PPO update
            logger.debug("Performing PPO update...")
            metrics = trainer.ppo_update(
                prompts=posts,
                generated_texts=summaries,
                old_log_probs=old_log_probs,
                rewards=rewards,
                optimizer=optimizer,
                max_ppo_epochs=num_ppo_epochs
            )
            
            global_step += 1
            
            # Log metrics
            if use_wandb:
                wandb.log({
                    "step": global_step,
                    "ppo/loss": metrics["loss"],
                    "ppo/kl": metrics["kl"],
                    "ppo/clip_fraction": metrics["clip_fraction"],
                    "ppo/mean_reward": metrics["mean_reward"],
                    "ppo/mean_advantage": metrics["mean_advantage"]
                })
            
            progress_bar.set_postfix({
                "loss": metrics["loss"],
                "reward": metrics["mean_reward"],
                "kl": metrics["kl"]
            })
            
            # Save checkpoint periodically
            if global_step % config['training']['save_steps'] == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                policy_model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("PPO training complete!")
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train PPO model")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="./checkpoints/sft", help="SFT model path")
    parser.add_argument("--reward-model", type=str, default="./checkpoints/reward_model/best_model.pt", help="Reward model path")
    parser.add_argument("--train-data", type=str, default="./data/splits/train.json", help="Training data path")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/ppo", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_ppo(
        model_path=args.model,
        reward_model_path=args.reward_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        config=config,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()

