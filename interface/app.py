"""
Gradio Web Interface for Reddit Summarizer
Allows users to input Reddit posts, generate summaries, and provide feedback
"""

import os
import json
import argparse
import yaml
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging
from datetime import datetime

from models.sft_model import SFTModel
from models.reward_model import RewardModel
from data.fetch_reddit_data import RedditDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizerApp:
    """Gradio app for Reddit summarization with feedback collection"""
    
    def __init__(
        self,
        model_path: str,
        reward_model_path: Optional[str] = None,
        feedback_db_path: str = "./data/feedback/feedback.json"
    ):
        """
        Initialize the app
        
        Args:
            model_path: Path to trained model
            reward_model_path: Optional path to reward model
            feedback_db_path: Path to feedback database
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feedback_db_path = feedback_db_path
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = SFTModel(model_name=model_path, device=self.device)
        
        # Load reward model if provided
        self.reward_model = None
        if reward_model_path:
            logger.info(f"Loading reward model from {reward_model_path}")
            self.reward_model = RewardModel(device=self.device)
            self.reward_model.load(reward_model_path)
        
        # Initialize feedback database
        os.makedirs(os.path.dirname(feedback_db_path), exist_ok=True)
        if os.path.exists(feedback_db_path):
            with open(feedback_db_path, 'r') as f:
                self.feedback_db = json.load(f)
        else:
            self.feedback_db = []
        
        logger.info("App initialized successfully")
    
    def generate_summary(
        self,
        post_text: str,
        max_length: int = 128,
        temperature: float = 0.7,
        num_variations: int = 1
    ) -> List[str]:
        """
        Generate summary(ies) for a post
        
        Args:
            post_text: Post text
            max_length: Maximum summary length
            temperature: Sampling temperature
            num_variations: Number of summary variations to generate
            
        Returns:
            List of generated summaries
        """
        if not post_text.strip():
            return ["Please enter a Reddit post to summarize."]
        
        summaries = []
        for _ in range(num_variations):
            summary = self.model.generate_summary(
                post=post_text,
                max_length=max_length,
                temperature=temperature
            )
            summaries.append(summary)
        
        return summaries
    
    def generate_with_reward(
        self,
        post_text: str,
        max_length: int = 128,
        temperature: float = 0.7,
        num_variations: int = 3
    ) -> tuple:
        """
        Generate summaries with reward scores
        
        Args:
            post_text: Post text
            max_length: Maximum summary length
            temperature: Sampling temperature
            num_variations: Number of variations
            
        Returns:
            Tuple of (summaries, reward_scores)
        """
        summaries = self.generate_summary(
            post_text,
            max_length,
            temperature,
            num_variations
        )
        
        reward_scores = []
        if self.reward_model:
            for summary in summaries:
                reward = self.reward_model.compute_reward(
                    post_text,
                    summary,
                    self.model.tokenizer
                )
                reward_scores.append(f"{reward:.4f}")
        else:
            reward_scores = ["N/A"] * len(summaries)
        
        return summaries, reward_scores
    
    def save_feedback(
        self,
        post: str,
        summary: str,
        rating: int,
        comparison: Optional[str] = None
    ):
        """
        Save user feedback
        
        Args:
            post: Post text
            summary: Summary text
            rating: Rating (1-5 or -1/1 for thumbs)
            comparison: Optional comparison summary
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "post": post,
            "summary": summary,
            "rating": rating,
            "comparison": comparison
        }
        
        self.feedback_db.append(feedback_entry)
        
        # Save to file
        with open(self.feedback_db_path, 'w') as f:
            json.dump(self.feedback_db, f, indent=2)
        
        logger.info(f"Feedback saved: {len(self.feedback_db)} total entries")
        return f"Feedback saved! Total feedback entries: {len(self.feedback_db)}"
    
    def fetch_reddit_post(
        self,
        url: str,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = "RedditSummarizer/1.0"
    ) -> str:
        """
        Fetch Reddit post from URL
        
        Args:
            url: Reddit post URL
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
            
        Returns:
            Post text
        """
        if not url:
            return ""
        
        try:
            # Extract post ID from URL
            if "reddit.com" in url:
                # Parse Reddit URL
                parts = url.split("/")
                post_id = None
                for i, part in enumerate(parts):
                    if part == "comments" and i + 1 < len(parts):
                        post_id = parts[i + 1]
                        break
                
                if not post_id:
                    return "Could not parse Reddit URL. Please provide a valid Reddit post URL."
                
                # Fetch post (requires API credentials)
                if not client_id or not client_secret:
                    return "Reddit API credentials required. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
                
                fetcher = RedditDataFetcher(client_id, client_secret, user_agent)
                posts = fetcher.fetch_posts("", limit=1)  # This won't work directly, need to fetch by ID
                # For now, return a placeholder
                return "Reddit post fetching requires additional implementation. Please paste the post text directly."
            
            return ""
        except Exception as e:
            logger.error(f"Error fetching Reddit post: {e}")
            return f"Error fetching post: {str(e)}"
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Reddit Post Summarizer", theme=gr.themes.Soft()) as app:
            gr.Markdown("# Reddit Post Summarizer with RLHF")
            gr.Markdown("Enter a Reddit post or URL to generate summaries. Provide feedback to help improve the model!")
            
            with gr.Row():
                with gr.Column(scale=2):
                    post_input = gr.Textbox(
                        label="Reddit Post",
                        placeholder="Paste your Reddit post text here, or enter a Reddit URL...",
                        lines=10,
                        value=""
                    )
                    
                    with gr.Row():
                        fetch_btn = gr.Button("Fetch from URL", variant="secondary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                    
                    with gr.Accordion("Generation Settings", open=False):
                        max_length = gr.Slider(
                            minimum=50,
                            maximum=256,
                            value=128,
                            step=10,
                            label="Max Summary Length"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        num_variations = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of Variations"
                        )
                    
                    generate_btn = gr.Button("Generate Summary", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    output_summaries = gr.Textbox(
                        label="Generated Summaries",
                        lines=10,
                        value=""
                    )
                    
                    reward_scores = gr.Textbox(
                        label="Reward Scores",
                        lines=3,
                        value=""
                    )
                    
                    with gr.Accordion("Feedback", open=True):
                        feedback_rating = gr.Radio(
                            choices=[("👍 Good", 1), ("👎 Bad", -1)],
                            label="Rate this summary",
                            value=None
                        )
                        
                        feedback_btn = gr.Button("Submit Feedback", variant="secondary")
                        feedback_output = gr.Textbox(label="Feedback Status", interactive=False)
            
            # Example section
            with gr.Accordion("Examples", open=False):
                examples = [
                    [
                        "I've been working on this machine learning project for months. The main challenge was dealing with imbalanced datasets. I tried various techniques like SMOTE, class weighting, and ensemble methods. Finally, I found that a combination of focal loss and data augmentation worked best. The model now achieves 95% accuracy on the test set!"
                    ],
                    [
                        "Just finished reading 'The Three-Body Problem' by Liu Cixin. What an incredible book! The science is mind-bending, the characters are complex, and the plot twists are unexpected. I especially loved how it combines hard science fiction with deep philosophical questions about humanity and the universe. Highly recommend to anyone interested in sci-fi."
                    ],
                    [
                        "After years of procrastination, I finally started learning to play the guitar. I'm using online tutorials and practicing 30 minutes every day. It's frustrating at first, but I'm starting to see progress. My fingers are getting stronger and I can now play a few basic chords. Any tips for a beginner?"
                    ]
                ]
                gr.Examples(examples=examples, inputs=post_input)
            
            # Event handlers
            def generate_summaries(post, max_len, temp, num_var):
                if not post.strip():
                    return "Please enter a post to summarize.", ""
                
                summaries, scores = self.generate_with_reward(
                    post,
                    max_len,
                    temp,
                    num_var
                )
                
                summary_text = "\n\n".join([f"Summary {i+1}:\n{s}" for i, s in enumerate(summaries)])
                score_text = "\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(scores)])
                
                return summary_text, score_text
            
            def submit_feedback(post, summary_text, rating):
                if not post or not summary_text or rating is None:
                    return "Please provide all feedback information."
                
                # Extract first summary if multiple
                summaries = summary_text.split("\n\n")
                first_summary = summaries[0].split(":\n")[-1] if ":\n" in summaries[0] else summaries[0]
                
                result = self.save_feedback(post, first_summary, rating)
                return result
            
            generate_btn.click(
                fn=generate_summaries,
                inputs=[post_input, max_length, temperature, num_variations],
                outputs=[output_summaries, reward_scores]
            )
            
            feedback_btn.click(
                fn=submit_feedback,
                inputs=[post_input, output_summaries, feedback_rating],
                outputs=[feedback_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "", None, ""),
                outputs=[post_input, output_summaries, reward_scores, feedback_rating, feedback_output]
            )
        
        return app


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio app for Reddit summarizer")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="./checkpoints/sft", help="Model checkpoint path")
    parser.add_argument("--reward-model", type=str, default=None, help="Reward model path (optional)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    # Load config if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if args.reward_model is None and 'reward_model' in config.get('paths', {}):
            args.reward_model = config['paths'].get('reward_model')
    
    # Initialize app
    app_instance = SummarizerApp(
        model_path=args.model,
        reward_model_path=args.reward_model
    )
    
    # Create interface
    app = app_instance.create_interface()
    
    # Launch
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()

