"""
Setup script to create necessary directories
"""

import os
from pathlib import Path

def setup_directories():
    """Create all necessary directories for the project"""
    directories = [
        "data/raw",
        "data/processed",
        "data/splits",
        "data/feedback",
        "checkpoints/sft",
        "checkpoints/reward_model",
        "checkpoints/ppo",
        "outputs",
        "logs",
        "runs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nAll directories created successfully!")

if __name__ == "__main__":
    setup_directories()

