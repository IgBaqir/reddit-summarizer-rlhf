"""
Quick demo script to test the summarization without full training
Uses a pre-trained GPT-2 model directly
"""

import sys

def demo_with_pretrained():
    """Demo using pre-trained GPT-2 without fine-tuning"""
    try:
        print("Loading pre-trained GPT-2 model for demo...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded on {device}")
        print("\n" + "="*60)
        print("DEMO: Reddit Post Summarization")
        print("="*60)
        
        # Test post
        test_post = """I've been working on this machine learning project for months and finally got it working. 
The key was understanding the data flow and making sure all the components communicate properly. 
I tried various techniques like SMOTE for handling imbalanced data, class weighting, and ensemble methods. 
Finally, I found that a combination of focal loss and data augmentation worked best. 
The model now achieves 95% accuracy on the test set!"""
        
        print(f"\n📝 Original Post:")
        print("-" * 60)
        print(test_post)
        print("-" * 60)
        
        # Generate summary
        input_text = f"{test_post} {tokenizer.eos_token} Summary:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        print("\n🤖 Generating summary...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = generated_text[len(input_text):].strip()
        
        print("\n📄 Generated Summary:")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        
        print("\n✅ Demo complete!")
        print("\nNote: This is using a base GPT-2 model without fine-tuning.")
        print("For better results, train the model on Reddit data first.")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip3 install torch transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_with_pretrained()

