#!/bin/bash

# Script to push Reddit Summarizer RLHF to GitHub

echo "=========================================="
echo "Push to GitHub - Reddit Summarizer RLHF"
echo "=========================================="
echo ""

# Check if remote exists
if git remote | grep -q origin; then
    echo "✓ Remote 'origin' already exists"
    git remote -v
    echo ""
    echo "To push, run:"
    echo "  git push -u origin main"
else
    echo "No remote repository configured yet."
    echo ""
    echo "To push to GitHub, follow these steps:"
    echo ""
    echo "1. Create a new repository on GitHub:"
    echo "   https://github.com/new"
    echo ""
    echo "2. Then run one of these commands:"
    echo ""
    echo "   For HTTPS:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo ""
    echo "   For SSH:"
    echo "   git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git"
    echo ""
    echo "3. Push to GitHub:"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
fi

echo ""
echo "Current status:"
git status --short | head -10

