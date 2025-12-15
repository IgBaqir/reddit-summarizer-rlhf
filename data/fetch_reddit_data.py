"""
Reddit Data Collection Module
Fetches posts and comments from Reddit using PRAW library
"""

import praw
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditDataFetcher:
    """Class to fetch Reddit posts and comments using PRAW"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API client
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for Reddit API
        """
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("Reddit API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            raise
    
    def fetch_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort_by: str = "hot",
        min_length: int = 200,
        max_length: int = 2000
    ) -> List[Dict]:
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to fetch
            sort_by: Sort method ('hot', 'new', 'top', 'rising')
            min_length: Minimum post text length
            max_length: Maximum post text length
            
        Returns:
            List of post dictionaries
        """
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sort method
            if sort_by == "hot":
                submissions = subreddit.hot(limit=limit)
            elif sort_by == "new":
                submissions = subreddit.new(limit=limit)
            elif sort_by == "top":
                submissions = subreddit.top(limit=limit, time_filter="week")
            elif sort_by == "rising":
                submissions = subreddit.rising(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)
            
            for submission in submissions:
                try:
                    # Skip stickied posts
                    if submission.stickied:
                        continue
                    
                    # Get post text (selftext for self posts, title for link posts)
                    post_text = submission.selftext if submission.selftext else submission.title
                    
                    # Filter by length
                    if len(post_text) < min_length or len(post_text) > max_length:
                        continue
                    
                    post_data = {
                        "id": submission.id,
                        "title": submission.title,
                        "text": post_text,
                        "full_text": f"{submission.title}\n\n{submission.selftext}",
                        "subreddit": subreddit_name,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created_utc": submission.created_utc,
                        "url": submission.url,
                        "permalink": f"https://reddit.com{submission.permalink}",
                        "author": str(submission.author) if submission.author else "[deleted]"
                    }
                    
                    posts.append(post_data)
                    logger.debug(f"Fetched post: {submission.id}")
                    
                except Exception as e:
                    logger.warning(f"Error processing post {submission.id}: {e}")
                    continue
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            raise
        
        return posts
    
    def fetch_comments(
        self,
        submission_id: str,
        max_comments: int = 10,
        min_length: int = 50
    ) -> List[Dict]:
        """
        Fetch top-level comments from a post
        
        Args:
            submission_id: Reddit post ID
            max_comments: Maximum number of comments to fetch
            min_length: Minimum comment length
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" placeholders
            
            for comment in submission.comments.list()[:max_comments]:
                try:
                    if hasattr(comment, 'body') and len(comment.body) >= min_length:
                        comment_data = {
                            "id": comment.id,
                            "body": comment.body,
                            "score": comment.score,
                            "created_utc": comment.created_utc,
                            "author": str(comment.author) if comment.author else "[deleted]"
                        }
                        comments.append(comment_data)
                except Exception as e:
                    logger.warning(f"Error processing comment {comment.id}: {e}")
                    continue
            
            logger.debug(f"Fetched {len(comments)} comments for post {submission_id}")
            
        except Exception as e:
            logger.warning(f"Error fetching comments for post {submission_id}: {e}")
        
        return comments
    
    def fetch_subreddit_data(
        self,
        subreddit_name: str,
        posts_per_subreddit: int = 100,
        max_comments_per_post: int = 10,
        min_post_length: int = 200,
        max_post_length: int = 2000
    ) -> List[Dict]:
        """
        Fetch complete data (posts + comments) from a subreddit
        
        Args:
            subreddit_name: Name of the subreddit
            posts_per_subreddit: Number of posts to fetch
            max_comments_per_post: Maximum comments per post
            min_post_length: Minimum post length
            max_post_length: Maximum post length
            
        Returns:
            List of dictionaries containing post and comment data
        """
        all_data = []
        posts = self.fetch_posts(
            subreddit_name=subreddit_name,
            limit=posts_per_subreddit,
            min_length=min_post_length,
            max_length=max_post_length
        )
        
        for post in posts:
            comments = self.fetch_comments(
                submission_id=post["id"],
                max_comments=max_comments_per_post
            )
            
            post["comments"] = comments
            all_data.append(post)
            
            # Rate limiting - be respectful to Reddit API
            time.sleep(0.5)
        
        logger.info(f"Fetched {len(all_data)} posts with comments from r/{subreddit_name}")
        return all_data
    
    def save_data(self, data: List[Dict], output_path: str):
        """
        Save fetched data to JSON file
        
        Args:
            data: List of data dictionaries
            output_path: Path to save JSON file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data)} items to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")
            raise


def main():
    """Main function to fetch Reddit data"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Fetch Reddit posts and comments")
    parser.add_argument("--subreddit", type=str, default="AskReddit", help="Subreddit name")
    parser.add_argument("--limit", type=int, default=100, help="Number of posts to fetch")
    parser.add_argument("--output", type=str, default="./data/raw/reddit_data.json", help="Output file path")
    parser.add_argument("--client-id", type=str, help="Reddit API client ID")
    parser.add_argument("--client-secret", type=str, help="Reddit API client secret")
    parser.add_argument("--user-agent", type=str, default="RedditSummarizer/1.0", help="User agent string")
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    client_id = args.client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = args.user_agent or os.getenv("REDDIT_USER_AGENT", "RedditSummarizer/1.0")
    
    if not client_id or not client_secret:
        raise ValueError("Reddit API credentials not provided. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
    
    # Initialize fetcher
    fetcher = RedditDataFetcher(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    # Fetch data
    data = fetcher.fetch_subreddit_data(
        subreddit_name=args.subreddit,
        posts_per_subreddit=args.limit,
        max_comments_per_post=10,
        min_post_length=200,
        max_post_length=2000
    )
    
    # Save data
    fetcher.save_data(data, args.output)
    
    print(f"Successfully fetched and saved {len(data)} posts to {args.output}")


if __name__ == "__main__":
    main()

