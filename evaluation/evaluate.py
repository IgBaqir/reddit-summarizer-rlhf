"""
Evaluation Module
Evaluates models using ROUGE scores and other metrics
"""

import os
import argparse
import json
import yaml
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from rouge_score import rouge_scorer
from data.dataset import load_data_from_json
from models.reward_model import RewardModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Evaluator:
    """Evaluator for summarization models"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        reward_model_path: str = None,
        device: str = None
    ):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            reward_model_path: Optional path to reward model
            device: Device to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load reward model if provided
        self.reward_model = None
        if reward_model_path:
            logger.info(f"Loading reward model from {reward_model_path}")
            self.reward_model = RewardModel(device=self.device)
            self.reward_model.load(reward_model_path)
            self.reward_model.eval()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def generate_summary(
        self,
        post: str,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate summary for a post
        
        Args:
            post: Post text
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated summary
        """
        input_text = f"{post} {self.tokenizer.eos_token}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def compute_rouge_scores(
        self,
        reference: str,
        generated: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores
        
        Args:
            reference: Reference summary
            generated: Generated summary
            
        Returns:
            Dictionary of ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)
        
        # Convert to dictionary format
        result = {}
        for metric, score in scores.items():
            result[metric] = {
                "precision": score.precision,
                "recall": score.recall,
                "fmeasure": score.fmeasure
            }
        
        return result
    
    def compute_reward_score(
        self,
        post: str,
        summary: str
    ) -> float:
        """
        Compute reward score for a summary
        
        Args:
            post: Post text
            summary: Summary text
            
        Returns:
            Reward score
        """
        if self.reward_model is None:
            return 0.0
        
        return self.reward_model.compute_reward(post, summary, self.tokenizer)
    
    def evaluate(
        self,
        test_data: List[Dict],
        max_samples: int = None
    ) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_data: List of test examples
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if max_samples:
            test_data = test_data[:max_samples]
        
        logger.info(f"Evaluating on {len(test_data)} samples...")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        reward_scores = []
        
        for i, item in enumerate(test_data):
            post = item["post"]
            reference_summary = item["summary"]
            
            # Generate summary
            generated_summary = self.generate_summary(post)
            
            # Compute ROUGE scores
            rouge_scores = self.compute_rouge_scores(reference_summary, generated_summary)
            rouge1_scores.append(rouge_scores["rouge1"]["fmeasure"])
            rouge2_scores.append(rouge_scores["rouge2"]["fmeasure"])
            rougeL_scores.append(rouge_scores["rougeL"]["fmeasure"])
            
            # Compute reward score
            reward = self.compute_reward_score(post, generated_summary)
            reward_scores.append(reward)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_data)} samples")
        
        # Compute averages
        results = {
            "rouge1": {
                "mean": sum(rouge1_scores) / len(rouge1_scores),
                "scores": rouge1_scores
            },
            "rouge2": {
                "mean": sum(rouge2_scores) / len(rouge2_scores),
                "scores": rouge2_scores
            },
            "rougeL": {
                "mean": sum(rougeL_scores) / len(rougeL_scores),
                "scores": rougeL_scores
            },
            "reward": {
                "mean": sum(reward_scores) / len(reward_scores),
                "scores": reward_scores
            },
            "num_samples": len(test_data)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization model")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--test-data", type=str, default="./data/splits/test.json", help="Test data path")
    parser.add_argument("--reward-model", type=str, default=None, help="Reward model path (optional)")
    parser.add_argument("--output", type=str, default="./outputs/evaluation_results.json", help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load test data
    test_data = load_data_from_json(args.test_data)
    
    # Initialize evaluator
    evaluator = Evaluator(
        model_path=args.model,
        tokenizer_path=args.model,  # Tokenizer should be in same directory
        reward_model_path=args.reward_model,
        device=None
    )
    
    # Evaluate
    results = evaluator.evaluate(test_data, max_samples=args.max_samples)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"ROUGE-1 F1: {results['rouge1']['mean']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2']['mean']:.4f}")
    print(f"ROUGE-L F1: {results['rougeL']['mean']:.4f}")
    if results['reward']['mean'] != 0.0:
        print(f"Mean Reward: {results['reward']['mean']:.4f}")
    print(f"Number of samples: {results['num_samples']}")
    print("="*50)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

