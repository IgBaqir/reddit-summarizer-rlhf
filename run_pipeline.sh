#!/bin/bash

# Complete RLHF Training Pipeline Script
# This script runs the entire training pipeline from data collection to PPO training

set -e  # Exit on error

echo "=========================================="
echo "Reddit Summarizer RLHF - Training Pipeline"
echo "=========================================="

# Configuration
SUBREDDIT="AskReddit"
POSTS_LIMIT=100
CONFIG_PATH="./configs/config.yaml"
BASE_MODEL="gpt2"

# Step 1: Setup directories
echo ""
echo "Step 1: Setting up directories..."
python setup_directories.py

# Step 2: Fetch Reddit data
echo ""
echo "Step 2: Fetching Reddit data..."
if [ -z "$REDDIT_CLIENT_ID" ] || [ -z "$REDDIT_CLIENT_SECRET" ]; then
    echo "Warning: Reddit API credentials not set. Skipping data collection."
    echo "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
    echo "Or run manually: python data/fetch_reddit_data.py --subreddit $SUBREDDIT --limit $POSTS_LIMIT"
else
    python data/fetch_reddit_data.py \
        --subreddit $SUBREDDIT \
        --limit $POSTS_LIMIT \
        --output ./data/raw/reddit_data.json
fi

# Step 3: Preprocess data
echo ""
echo "Step 3: Preprocessing data..."
if [ -f "./data/raw/reddit_data.json" ]; then
    python data/preprocess.py \
        --input ./data/raw/reddit_data.json \
        --output ./data/processed/processed_data.json
else
    echo "Warning: No raw data found. Skipping preprocessing."
    echo "Please run data collection first or provide processed data."
fi

# Step 4: Create train/val/test splits
echo ""
echo "Step 4: Creating data splits..."
if [ -f "./data/processed/processed_data.json" ]; then
    python -c "
from data.dataset import load_data_from_json, split_data, save_splits
data = load_data_from_json('./data/processed/processed_data.json')
train, val, test = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
save_splits(train, val, test, './data/splits')
print('Data splits created successfully!')
"
else
    echo "Warning: No processed data found. Skipping split creation."
fi

# Step 5: Supervised Fine-Tuning
echo ""
echo "Step 5: Training SFT model..."
if [ -f "./data/splits/train.json" ] && [ -f "./data/splits/val.json" ]; then
    python training/train_sft.py \
        --config $CONFIG_PATH \
        --model $BASE_MODEL \
        --train-data ./data/splits/train.json \
        --val-data ./data/splits/val.json \
        --output-dir ./checkpoints/sft
else
    echo "Warning: Training data not found. Skipping SFT training."
fi

# Step 6: Train Reward Model
echo ""
echo "Step 6: Training reward model..."
if [ -f "./data/splits/train.json" ] && [ -f "./data/splits/val.json" ]; then
    python training/train_reward.py \
        --config $CONFIG_PATH \
        --model $BASE_MODEL \
        --train-data ./data/splits/train.json \
        --val-data ./data/splits/val.json \
        --output-dir ./checkpoints/reward_model
else
    echo "Warning: Training data not found. Skipping reward model training."
fi

# Step 7: PPO Training
echo ""
echo "Step 7: PPO training..."
if [ -d "./checkpoints/sft" ] && [ -f "./checkpoints/reward_model/best_model.pt" ]; then
    python training/train_ppo.py \
        --config $CONFIG_PATH \
        --model ./checkpoints/sft \
        --reward-model ./checkpoints/reward_model/best_model.pt \
        --train-data ./data/splits/train.json \
        --output-dir ./checkpoints/ppo
else
    echo "Warning: SFT or reward model not found. Skipping PPO training."
fi

# Step 8: Evaluation
echo ""
echo "Step 8: Evaluating model..."
if [ -d "./checkpoints/ppo" ] && [ -f "./data/splits/test.json" ]; then
    python evaluation/evaluate.py \
        --config $CONFIG_PATH \
        --model ./checkpoints/ppo \
        --test-data ./data/splits/test.json \
        --reward-model ./checkpoints/reward_model/best_model.pt \
        --output ./outputs/evaluation_results.json
else
    echo "Warning: Model or test data not found. Skipping evaluation."
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "To launch the web interface, run:"
echo "python interface/app.py --model ./checkpoints/ppo --reward-model ./checkpoints/reward_model/best_model.pt"

