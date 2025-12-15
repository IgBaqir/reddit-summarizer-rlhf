"""
Dataset Utilities for Reddit Summarization
Creates PyTorch datasets for SFT, reward model, and PPO training
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """Dataset for supervised fine-tuning on summarization"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128
    ):
        """
        Initialize summarization dataset
        
        Args:
            data: List of dictionaries with 'post' and 'summary' keys
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Add special tokens if not already present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        post = item["post"]
        summary = item["summary"]
        
        # Tokenize input (post)
        input_encoding = self.tokenizer(
            post,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize output (summary)
        output_encoding = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For causal LM, we concatenate input and output
        # Format: <post> <eos> <summary> <eos>
        full_text = f"{post} {self.tokenizer.eos_token} {summary} {self.tokenizer.eos_token}"
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_input_length + self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": full_encoding["input_ids"].squeeze(),
            "attention_mask": full_encoding["attention_mask"].squeeze(),
            "labels": full_encoding["input_ids"].squeeze(),  # For causal LM, labels = input_ids
            "post": post,
            "summary": summary
        }


class PreferenceDataset(Dataset):
    """Dataset for reward model training on preference comparisons"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """
        Initialize preference dataset
        
        Args:
            data: List of dictionaries with 'post', 'summary_chosen', 'summary_rejected' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        post = item["post"]
        summary_chosen = item["summary_chosen"]
        summary_rejected = item["summary_rejected"]
        
        # Create prompt for chosen summary
        chosen_text = f"{post} {self.tokenizer.eos_token} {summary_chosen}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create prompt for rejected summary
        rejected_text = f"{post} {self.tokenizer.eos_token} {summary_rejected}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids_chosen": chosen_encoding["input_ids"].squeeze(),
            "attention_mask_chosen": chosen_encoding["attention_mask"].squeeze(),
            "input_ids_rejected": rejected_encoding["input_ids"].squeeze(),
            "attention_mask_rejected": rejected_encoding["attention_mask"].squeeze(),
            "post": post,
            "summary_chosen": summary_chosen,
            "summary_rejected": summary_rejected
        }


class PPODataset(Dataset):
    """Dataset for PPO training"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128
    ):
        """
        Initialize PPO dataset
        
        Args:
            data: List of dictionaries with 'post' key
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        post = item["post"]
        
        # Tokenize input (post)
        input_encoding = self.tokenizer(
            post,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "post": post
        }


def load_data_from_json(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets
    
    Args:
        data: List of data dictionaries
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    n = len(data_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = data_shuffled[:n_train]
    val_data = data_shuffled[n_train:n_train + n_val]
    test_data = data_shuffled[n_train + n_val:]
    
    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def create_preference_pairs(
    data: List[Dict],
    num_pairs: Optional[int] = None
) -> List[Dict]:
    """
    Create preference pairs from data with multiple summaries
    
    Args:
        data: List of dictionaries with 'post' and 'summaries' keys
        num_pairs: Number of pairs to create (None = use all possible pairs)
        
    Returns:
        List of preference pair dictionaries
    """
    preference_pairs = []
    
    for item in data:
        summaries = item.get("summaries", [])
        if len(summaries) < 2:
            continue
        
        # Use first summary as chosen (assuming it's the best one)
        # Create pairs with other summaries as rejected
        chosen = summaries[0]
        for rejected in summaries[1:]:
            preference_pairs.append({
                "post": item["post"],
                "summary_chosen": chosen,
                "summary_rejected": rejected
            })
    
    # Shuffle and optionally limit
    random.shuffle(preference_pairs)
    if num_pairs is not None:
        preference_pairs = preference_pairs[:num_pairs]
    
    logger.info(f"Created {len(preference_pairs)} preference pairs")
    
    return preference_pairs


def save_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str
):
    """Save train/val/test splits to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--input", type=str, default="./data/processed/processed_data.json")
    parser.add_argument("--output-dir", type=str, default="./data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    data = load_data_from_json(args.input)
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    save_splits(train_data, val_data, test_data, args.output_dir)

