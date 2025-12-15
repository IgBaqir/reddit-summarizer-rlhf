#!/bin/bash

# Quick start script for the Reddit Summarizer RLHF project

echo "=========================================="
echo "Reddit Summarizer RLHF - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Create sample data if it doesn't exist
if [ ! -f "data/splits/train.json" ]; then
    echo ""
    echo "Creating sample data..."
    python3 create_sample_data_simple.py
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  1. Test demo:        python3 run_demo.py"
echo "  2. Train SFT:         python3 training/train_sft.py --no-wandb"
echo "  3. Train Reward:     python3 training/train_reward.py --no-wandb"
echo "  4. Train PPO:        python3 training/train_ppo.py --no-wandb"
echo "  5. Launch Interface: python3 interface/app.py --model gpt2"
echo ""
echo "Note: Use '--no-wandb' if you don't have wandb configured"
echo ""

