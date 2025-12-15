# Reddit Post Summarizer with RLHF

A complete system for summarizing Reddit posts using Reinforcement Learning from Human Feedback (RLHF), implementing Supervised Fine-Tuning (SFT), Reward Model training, and Proximal Policy Optimization (PPO).

## 🎯 Project Overview

This project implements a full RLHF pipeline for training a Reddit post summarization model:

1. **Data Collection**: Fetches Reddit posts and comments using PRAW
2. **Supervised Fine-Tuning (SFT)**: Fine-tunes a base language model on post-summary pairs
3. **Reward Model Training**: Trains a model to score summaries based on human preferences
4. **PPO Training**: Optimizes the policy model using Proximal Policy Optimization
5. **Evaluation**: Comprehensive evaluation using ROUGE scores and reward metrics
6. **Web Interface**: Interactive Gradio interface for summarization and feedback collection

## 📁 Project Structure

```
reddit-summarizer-rlhf/
├── data/
│   ├── fetch_reddit_data.py    # Reddit API data collection
│   ├── preprocess.py           # Data preprocessing and cleaning
│   └── dataset.py              # PyTorch dataset utilities
├── models/
│   ├── sft_model.py            # Supervised fine-tuning model
│   ├── reward_model.py         # Reward model architecture
│   └── ppo_trainer.py          # PPO trainer implementation
├── training/
│   ├── train_sft.py            # SFT training script
│   ├── train_reward.py         # Reward model training script
│   └── train_ppo.py            # PPO training script
├── evaluation/
│   └── evaluate.py             # Evaluation with ROUGE metrics
├── interface/
│   └── app.py                  # Gradio web interface
├── configs/
│   └── config.yaml             # Configuration file
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd reddit-summarizer-rlhf

# Install dependencies
pip install -r requirements.txt
```

### 2. Reddit API Setup

Create a `.env` file in the project root with your Reddit API credentials:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=RedditSummarizer/1.0
```

To get Reddit API credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" or "create app"
3. Select "script" as the app type
4. Copy the client ID (under the app name) and client secret

### 3. Data Collection

```bash
# Fetch Reddit posts from a subreddit
python data/fetch_reddit_data.py \
    --subreddit AskReddit \
    --limit 100 \
    --output ./data/raw/reddit_data.json

# Preprocess the data
python data/preprocess.py \
    --input ./data/raw/reddit_data.json \
    --output ./data/processed/processed_data.json

# Create train/val/test splits
python data/dataset.py \
    --input ./data/processed/processed_data.json \
    --output-dir ./data/splits
```

### 4. Training Pipeline

#### Step 1: Supervised Fine-Tuning (SFT)

```bash
python training/train_sft.py \
    --config ./configs/config.yaml \
    --model gpt2 \
    --train-data ./data/splits/train.json \
    --val-data ./data/splits/val.json \
    --output-dir ./checkpoints/sft
```

#### Step 2: Train Reward Model

```bash
python training/train_reward.py \
    --config ./configs/config.yaml \
    --model gpt2 \
    --train-data ./data/splits/train.json \
    --val-data ./data/splits/val.json \
    --output-dir ./checkpoints/reward_model
```

#### Step 3: PPO Training

```bash
python training/train_ppo.py \
    --config ./configs/config.yaml \
    --model ./checkpoints/sft \
    --reward-model ./checkpoints/reward_model/best_model.pt \
    --train-data ./data/splits/train.json \
    --output-dir ./checkpoints/ppo
```

### 5. Evaluation

```bash
python evaluation/evaluate.py \
    --config ./configs/config.yaml \
    --model ./checkpoints/ppo \
    --test-data ./data/splits/test.json \
    --reward-model ./checkpoints/reward_model/best_model.pt \
    --output ./outputs/evaluation_results.json
```

### 6. Launch Web Interface

```bash
python interface/app.py \
    --model ./checkpoints/ppo \
    --reward-model ./checkpoints/reward_model/best_model.pt \
    --port 7860
```

Then open your browser to `http://localhost:7860`

## 📊 Configuration

Edit `configs/config.yaml` to customize:

- **Model settings**: Base model, max lengths, generation parameters
- **Training hyperparameters**: Learning rates, batch sizes, epochs
- **PPO settings**: Clip epsilon, KL coefficient, GAE parameters
- **Data settings**: Subreddits, post limits, length constraints
- **Logging**: Wandb project, tensorboard settings

## 🔧 Key Features

### Data Pipeline
- **Reddit API Integration**: Fetch posts and comments from any subreddit
- **Data Preprocessing**: Clean text, extract summaries from comments
- **Dataset Splitting**: Automatic train/val/test splits
- **Preference Pair Creation**: Generate comparison pairs for reward model

### Models
- **SFT Model**: Fine-tuned language model for summarization
- **Reward Model**: Scores summaries based on human preferences
- **PPO Trainer**: Implements PPO algorithm with KL penalty

### Training
- **Mixed Precision**: FP16 training for faster iteration
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Checkpointing**: Save and resume training
- **Wandb Integration**: Track experiments and metrics

### Evaluation
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L metrics
- **Reward Scores**: Evaluate using trained reward model
- **Comprehensive Metrics**: Track multiple evaluation dimensions

### Web Interface
- **Interactive Summarization**: Generate summaries with adjustable parameters
- **Multiple Variations**: Generate and compare multiple summary candidates
- **Feedback Collection**: Collect user ratings for model improvement
- **Reward Scores**: Display reward model scores for each summary

## 📈 Training Pipeline Details

### 1. Supervised Fine-Tuning (SFT)
- Fine-tunes base model (GPT-2) on post-summary pairs
- Uses causal language modeling objective
- Creates initial summarization capability

### 2. Reward Model Training
- Trains on preference comparisons (chosen vs rejected summaries)
- Uses binary preference loss
- Learns to score summaries based on quality

### 3. PPO Training
- Optimizes policy model using reward model scores
- Implements KL divergence penalty to prevent model collapse
- Uses Generalized Advantage Estimation (GAE)
- Clips policy updates for stability

## 🎨 Web Interface Features

- **Post Input**: Paste Reddit posts or enter URLs
- **Generation Settings**: Adjustable length, temperature, number of variations
- **Summary Display**: View generated summaries with reward scores
- **Feedback System**: Rate summaries (thumbs up/down)
- **Feedback Storage**: All feedback saved for reward model retraining

## 📝 Usage Examples

### Fetch Data from Multiple Subreddits

```python
from data.fetch_reddit_data import RedditDataFetcher

fetcher = RedditDataFetcher(client_id, client_secret, user_agent)
data = fetcher.fetch_subreddit_data("AskReddit", posts_per_subreddit=100)
```

### Generate Summary

```python
from models.sft_model import SFTModel

model = SFTModel(model_name="./checkpoints/sft")
summary = model.generate_summary(post_text, max_length=128)
```

### Evaluate Model

```python
from evaluation.evaluate import Evaluator

evaluator = Evaluator(
    model_path="./checkpoints/ppo",
    tokenizer_path="./checkpoints/ppo",
    reward_model_path="./checkpoints/reward_model/best_model.pt"
)
results = evaluator.evaluate(test_data)
```

## 🐛 Troubleshooting

### Reddit API Rate Limiting
- The fetcher includes rate limiting (0.5s delay between requests)
- If you hit rate limits, reduce `posts_per_subreddit` or add longer delays

### Out of Memory Errors
- Reduce batch size in `config.yaml`
- Use gradient accumulation to maintain effective batch size
- Enable FP16 training
- Use smaller models (GPT-2 small instead of medium/large)

### Model Not Generating Good Summaries
- Ensure sufficient training data (at least 1000 examples)
- Check that summaries in training data are high quality
- Adjust temperature and top_p parameters
- Train for more epochs

## 🔮 Future Improvements

- [ ] Support for larger models (GPT-2 medium/large, GPT-Neo)
- [ ] Integration with OpenRLHF framework
- [ ] Multi-GPU training support
- [ ] Advanced reward model architectures
- [ ] Human-in-the-loop feedback collection
- [ ] Real-time model updates from feedback
- [ ] Support for different summary styles (bullet points, paragraphs)
- [ ] Export models to ONNX for faster inference

## 📚 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [RLHF Overview](https://huggingface.co/blog/rlhf)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

- This project uses Reddit's public API. Please respect Reddit's API terms of service.
- The models are trained on publicly available Reddit data.
- Generated summaries may not always be accurate or appropriate.
- Use responsibly and in accordance with Reddit's content policies.

