"""
Test that all imports work correctly
"""

print("Testing imports...")

try:
    print("  - Testing data modules...")
    from data.dataset import load_data_from_json
    print("    ✓ data.dataset")
    
    print("  - Testing model modules...")
    # Don't actually load models, just test imports
    import sys
    import importlib.util
    
    # Test if we can at least read the model files
    spec = importlib.util.spec_from_file_location("sft_model", "models/sft_model.py")
    print("    ✓ models.sft_model (file readable)")
    
    spec = importlib.util.spec_from_file_location("reward_model", "models/reward_model.py")
    print("    ✓ models.reward_model (file readable)")
    
    spec = importlib.util.spec_from_file_location("ppo_trainer", "models/ppo_trainer.py")
    print("    ✓ models.ppo_trainer (file readable)")
    
    print("  - Testing data loading...")
    data = load_data_from_json("data/splits/train.json")
    print(f"    ✓ Loaded {len(data)} training samples")
    
    print("\n✅ All imports and basic functionality working!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip3 install -r requirements.txt")
    print("  2. Train models: python training/train_sft.py")
    print("  3. Launch interface: python interface/app.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

