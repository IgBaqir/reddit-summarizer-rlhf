"""
Data Preprocessing Module
Cleans and processes Reddit posts for summarization training
"""

import json
import re
import os
from typing import List, Dict, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditDataPreprocessor:
    """Class to preprocess Reddit posts and comments"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.reddit_pattern = re.compile(r'r/\w+|/u/\w+')
        self.mention_pattern = re.compile(r'@\w+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, special formatting, etc.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Markdown links
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_main_content(self, post: Dict) -> str:
        """
        Extract main content from a Reddit post
        
        Args:
            post: Post dictionary
            
        Returns:
            Main content text
        """
        # Combine title and text
        title = post.get("title", "")
        text = post.get("text", "")
        
        if text:
            content = f"{title}\n\n{text}"
        else:
            content = title
        
        return self.clean_text(content)
    
    def create_summary_candidates(self, post: Dict, max_candidates: int = 3) -> List[str]:
        """
        Create summary candidates from post comments
        Uses top comments as potential summaries
        
        Args:
            post: Post dictionary with comments
            max_candidates: Maximum number of summary candidates
            
        Returns:
            List of summary candidate strings
        """
        candidates = []
        comments = post.get("comments", [])
        
        # Sort comments by score
        sorted_comments = sorted(comments, key=lambda x: x.get("score", 0), reverse=True)
        
        for comment in sorted_comments[:max_candidates]:
            body = comment.get("body", "")
            cleaned = self.clean_text(body)
            
            # Filter by length (comments that are too short or too long aren't good summaries)
            if 50 <= len(cleaned) <= 300:
                candidates.append(cleaned)
        
        return candidates
    
    def process_post(
        self,
        post: Dict,
        min_length: int = 200,
        max_length: int = 2000,
        min_summary_length: int = 20,
        max_summary_length: int = 128
    ) -> Dict:
        """
        Process a single post into training format
        
        Args:
            post: Raw post dictionary
            min_length: Minimum post length
            max_length: Maximum post length
            min_summary_length: Minimum summary length
            max_summary_length: Maximum summary length
            
        Returns:
            Processed post dictionary or None if invalid
        """
        # Extract and clean main content
        content = self.extract_main_content(post)
        
        # Filter by length
        if len(content) < min_length or len(content) > max_length:
            return None
        
        # Get summary candidates from comments
        summary_candidates = self.create_summary_candidates(post)
        
        # If no good candidates, create a simple summary from title
        if not summary_candidates:
            title = post.get("title", "")
            # Use first sentence or first 100 chars as fallback
            summary = title[:100] if len(title) > 100 else title
            summary_candidates = [summary]
        
        # Filter summaries by length
        valid_summaries = [
            s for s in summary_candidates
            if min_summary_length <= len(s) <= max_summary_length
        ]
        
        if not valid_summaries:
            return None
        
        processed = {
            "id": post.get("id"),
            "post": content,
            "title": post.get("title", ""),
            "summaries": valid_summaries,
            "summary": valid_summaries[0],  # Use best candidate as primary summary
            "subreddit": post.get("subreddit", ""),
            "metadata": {
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "url": post.get("permalink", "")
            }
        }
        
        return processed
    
    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        min_length: int = 200,
        max_length: int = 2000,
        min_summary_length: int = 20,
        max_summary_length: int = 128
    ) -> Tuple[List[Dict], Dict]:
        """
        Process entire dataset from JSON file
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to save processed JSON file
            min_length: Minimum post length
            max_length: Maximum post length
            min_summary_length: Minimum summary length
            max_summary_length: Maximum summary length
            
        Returns:
            Tuple of (processed_data, statistics)
        """
        logger.info(f"Loading data from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        stats = {
            "total": len(raw_data),
            "processed": 0,
            "skipped": 0,
            "avg_post_length": 0,
            "avg_summary_length": 0
        }
        
        total_post_length = 0
        total_summary_length = 0
        
        for post in raw_data:
            processed = self.process_post(
                post,
                min_length=min_length,
                max_length=max_length,
                min_summary_length=min_summary_length,
                max_summary_length=max_summary_length
            )
            
            if processed:
                processed_data.append(processed)
                stats["processed"] += 1
                total_post_length += len(processed["post"])
                total_summary_length += len(processed["summary"])
            else:
                stats["skipped"] += 1
        
        # Calculate averages
        if stats["processed"] > 0:
            stats["avg_post_length"] = total_post_length / stats["processed"]
            stats["avg_summary_length"] = total_summary_length / stats["processed"]
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed {stats['processed']} posts, skipped {stats['skipped']}")
        logger.info(f"Average post length: {stats['avg_post_length']:.1f}")
        logger.info(f"Average summary length: {stats['avg_summary_length']:.1f}")
        logger.info(f"Saved processed data to {output_path}")
        
        return processed_data, stats


def main():
    """Main function to preprocess Reddit data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Reddit data")
    parser.add_argument("--input", type=str, default="./data/raw/reddit_data.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="./data/processed/processed_data.json", help="Output JSON file")
    parser.add_argument("--min-length", type=int, default=200, help="Minimum post length")
    parser.add_argument("--max-length", type=int, default=2000, help="Maximum post length")
    parser.add_argument("--min-summary-length", type=int, default=20, help="Minimum summary length")
    parser.add_argument("--max-summary-length", type=int, default=128, help="Maximum summary length")
    
    args = parser.parse_args()
    
    preprocessor = RedditDataPreprocessor()
    processed_data, stats = preprocessor.process_dataset(
        input_path=args.input,
        output_path=args.output,
        min_length=args.min_length,
        max_length=args.max_length,
        min_summary_length=args.min_summary_length,
        max_summary_length=args.max_summary_length
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Processed: {stats['processed']}/{stats['total']} posts")
    print(f"Average post length: {stats['avg_post_length']:.1f} chars")
    print(f"Average summary length: {stats['avg_summary_length']:.1f} chars")


if __name__ == "__main__":
    main()

