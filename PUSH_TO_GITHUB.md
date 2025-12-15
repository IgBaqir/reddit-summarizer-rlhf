# Push to GitHub - Instructions

## Quick Commands

After creating a GitHub repository, run these commands:

```bash
# Add remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step-by-Step Guide

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `reddit-summarizer-rlhf` (or your preferred name)
3. Description: "Reddit Post Summarizer using Reinforcement Learning from Human Feedback (RLHF)"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 2. Add Remote and Push

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/reddit-summarizer-rlhf.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 3. Alternative: Using SSH

If you prefer SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/reddit-summarizer-rlhf.git
git branch -M main
git push -u origin main
```

## What's Included

The repository includes:
- ✅ All source code (data, models, training, evaluation, interface)
- ✅ Configuration files
- ✅ Documentation (README.md, QUICKSTART.md)
- ✅ Requirements.txt
- ✅ Sample data creation scripts
- ✅ Setup and run scripts

## What's Excluded (via .gitignore)

- ❌ `__pycache__/` - Python cache files
- ❌ `checkpoints/` - Model checkpoints (too large)
- ❌ `data/raw/`, `data/processed/`, `data/splits/` - Data files
- ❌ `outputs/` - Evaluation results
- ❌ `logs/`, `runs/` - Log files
- ❌ `venv/` - Virtual environment
- ❌ `.env` - Environment variables (contains secrets)

## Note on Large Files

Model checkpoints and data files are excluded. If you want to include them:
1. Use Git LFS: `git lfs install && git lfs track "*.pt" "*.pth" "*.bin"`
2. Or host them separately (HuggingFace Hub, Google Drive, etc.)

## Troubleshooting

### If push is rejected:
```bash
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### If you need to update later:
```bash
git add .
git commit -m "Your commit message"
git push
```

