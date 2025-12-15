# Quick Start Guide

This guide will help you get started with the Reddit Summarizer RLHF project quickly.

## Prerequisites

1. Python 3.8 or higher
2. Reddit API credentials (optional, for data collection)
3. CUDA-capable GPU (recommended for training)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup directories
python setup_directories.py
```

## Reddit API Setup (Optional)

If you want to collect your own Reddit data:

1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (select "script")
3. Copy your client ID and secret
4. Create a `.env` file:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=RedditSummarizer/1.0
```

## Quick Training Example

### Option 1: Use Provided Script

```bash
# Run the complete pipeline
./run_pipeline.sh
```

### Option 2: Step-by-Step

```bash
# 1. Collect data (if you have Reddit API credentials)
python data/fetch_reddit_data.py --subreddit AskReddit --limit 100

# 2. Preprocess
python data/preprocess.py

# 3. Create splits
python data/dataset.py

# 4. Train SFT
python training/train_sft.py

# 5. Train reward model
python training/train_reward.py

# 6. Train PPO
python training/train_ppo.py

# 7. Evaluate
python evaluation/evaluate.py --model ./checkpoints/ppo
```

## Using Pre-trained Models

If you have checkpoints, you can use them directly:

```python
from models.sft_model import SFTModel

# Load model
model = SFTModel(model_name="./checkpoints/sft")

# Generate summary
post = "Your Reddit post text here..."
summary = model.generate_summary(post, max_length=128)
print(summary)
```

## Launch Web Interface

```bash
python interface/app.py \
    --model ./checkpoints/ppo \
    --reward-model ./checkpoints/reward_model/best_model.pt \
    --port 7860
```

Then open http://localhost:7860 in your browser.

## Common Issues

### Out of Memory
- Reduce batch size in `configs/config.yaml`
- Use gradient accumulation
- Enable FP16 training

### No Reddit Data
- You can use any text data in the same format
- Or skip data collection and use preprocessed data

### Import Errors
- Make sure you're in the project root directory
- Install all requirements: `pip install -r requirements.txt`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize `configs/config.yaml` for your needs
- Collect feedback through the web interface
- Retrain reward model with collected feedback

