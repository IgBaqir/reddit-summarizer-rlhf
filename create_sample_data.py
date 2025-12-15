"""
Create sample data for testing the pipeline without Reddit API
"""

import json
import os

# Sample Reddit-style posts with summaries
sample_data = [
    {
        "id": "sample_1",
        "title": "I finally finished my machine learning project after 3 months",
        "text": "I've been working on this machine learning project for months and finally got it working. The key was understanding the data flow and making sure all the components communicate properly. I tried various techniques like SMOTE for handling imbalanced data, class weighting, and ensemble methods. Finally, I found that a combination of focal loss and data augmentation worked best. The model now achieves 95% accuracy on the test set!",
        "full_text": "I finally finished my machine learning project after 3 months\n\nI've been working on this machine learning project for months and finally got it working. The key was understanding the data flow and making sure all the components communicate properly. I tried various techniques like SMOTE for handling imbalanced data, class weighting, and ensemble methods. Finally, I found that a combination of focal loss and data augmentation worked best. The model now achieves 95% accuracy on the test set!",
        "subreddit": "MachineLearning",
        "score": 150,
        "num_comments": 25,
        "created_utc": 1234567890,
        "url": "https://reddit.com/r/MachineLearning/sample_1",
        "permalink": "https://reddit.com/r/MachineLearning/sample_1",
        "author": "test_user",
        "comments": [
            {
                "id": "comment_1",
                "body": "Great work! Focal loss is really effective for imbalanced datasets. The combination with data augmentation sounds like a solid approach.",
                "score": 45,
                "created_utc": 1234567900,
                "author": "ml_expert"
            },
            {
                "id": "comment_2",
                "body": "Congratulations on the 95% accuracy! That's impressive. Did you try any other loss functions before settling on focal loss?",
                "score": 32,
                "created_utc": 1234568000,
                "author": "curious_dev"
            }
        ]
    },
    {
        "id": "sample_2",
        "title": "Just finished reading 'The Three-Body Problem'",
        "text": "Just finished reading 'The Three-Body Problem' by Liu Cixin. What an incredible book! The science is mind-bending, the characters are complex, and the plot twists are unexpected. I especially loved how it combines hard science fiction with deep philosophical questions about humanity and the universe. The way it explores the dark forest hypothesis is fascinating. Highly recommend to anyone interested in sci-fi.",
        "full_text": "Just finished reading 'The Three-Body Problem'\n\nJust finished reading 'The Three-Body Problem' by Liu Cixin. What an incredible book! The science is mind-bending, the characters are complex, and the plot twists are unexpected. I especially loved how it combines hard science fiction with deep philosophical questions about humanity and the universe. The way it explores the dark forest hypothesis is fascinating. Highly recommend to anyone interested in sci-fi.",
        "subreddit": "books",
        "score": 200,
        "num_comments": 40,
        "created_utc": 1234568100,
        "url": "https://reddit.com/r/books/sample_2",
        "permalink": "https://reddit.com/r/books/sample_2",
        "author": "bookworm",
        "comments": [
            {
                "id": "comment_3",
                "body": "One of the best sci-fi books I've ever read. The trilogy gets even better!",
                "score": 78,
                "created_utc": 1234568200,
                "author": "sci_fi_fan"
            },
            {
                "id": "comment_4",
                "body": "The dark forest theory is mind-blowing. It really makes you think about first contact scenarios.",
                "score": 56,
                "created_utc": 1234568300,
                "author": "philosopher"
            }
        ]
    },
    {
        "id": "sample_3",
        "title": "Started learning guitar after years of procrastination",
        "text": "After years of procrastination, I finally started learning to play the guitar. I'm using online tutorials and practicing 30 minutes every day. It's frustrating at first, but I'm starting to see progress. My fingers are getting stronger and I can now play a few basic chords. The C, G, and D chords are becoming more natural. Any tips for a beginner?",
        "full_text": "Started learning guitar after years of procrastination\n\nAfter years of procrastination, I finally started learning to play the guitar. I'm using online tutorials and practicing 30 minutes every day. It's frustrating at first, but I'm starting to see progress. My fingers are getting stronger and I can now play a few basic chords. The C, G, and D chords are becoming more natural. Any tips for a beginner?",
        "subreddit": "guitar",
        "score": 120,
        "num_comments": 30,
        "created_utc": 1234568400,
        "url": "https://reddit.com/r/guitar/sample_3",
        "permalink": "https://reddit.com/r/guitar/sample_3",
        "author": "beginner_musician",
        "comments": [
            {
                "id": "comment_5",
                "body": "Keep practicing! Consistency is key. Try learning a simple song you love to stay motivated.",
                "score": 42,
                "created_utc": 1234568500,
                "author": "guitar_teacher"
            },
            {
                "id": "comment_6",
                "body": "Practice with a metronome. It will help your timing immensely as you progress.",
                "score": 38,
                "created_utc": 1234568600,
                "author": "pro_guitarist"
            }
        ]
    },
    {
        "id": "sample_4",
        "title": "My experience with intermittent fasting",
        "text": "I've been doing intermittent fasting for 6 months now, following a 16:8 schedule. The results have been amazing - I've lost 20 pounds, have more energy throughout the day, and my blood sugar levels have improved significantly. The first week was tough with hunger pangs, but my body adapted quickly. I usually eat between 12 PM and 8 PM, which works well with my work schedule.",
        "full_text": "My experience with intermittent fasting\n\nI've been doing intermittent fasting for 6 months now, following a 16:8 schedule. The results have been amazing - I've lost 20 pounds, have more energy throughout the day, and my blood sugar levels have improved significantly. The first week was tough with hunger pangs, but my body adapted quickly. I usually eat between 12 PM and 8 PM, which works well with my work schedule.",
        "subreddit": "fitness",
        "score": 180,
        "num_comments": 50,
        "created_utc": 1234568700,
        "url": "https://reddit.com/r/fitness/sample_4",
        "permalink": "https://reddit.com/r/fitness/sample_4",
        "author": "health_enthusiast",
        "comments": [
            {
                "id": "comment_7",
                "body": "Great results! Intermittent fasting combined with regular exercise has been a game changer for me too.",
                "score": 65,
                "created_utc": 1234568800,
                "author": "fitness_coach"
            },
            {
                "id": "comment_8",
                "body": "Make sure you're still getting enough nutrients during your eating window. Quality matters!",
                "score": 52,
                "created_utc": 1234568900,
                "author": "nutritionist"
            }
        ]
    },
    {
        "id": "sample_5",
        "title": "Built my first web app using React and Node.js",
        "text": "After months of learning JavaScript, I finally built my first full-stack web application! It's a task management app with user authentication, real-time updates, and a clean UI. I used React for the frontend, Node.js with Express for the backend, and MongoDB for the database. The hardest part was understanding state management and API design, but I'm really proud of what I created.",
        "full_text": "Built my first web app using React and Node.js\n\nAfter months of learning JavaScript, I finally built my first full-stack web application! It's a task management app with user authentication, real-time updates, and a clean UI. I used React for the frontend, Node.js with Express for the backend, and MongoDB for the database. The hardest part was understanding state management and API design, but I'm really proud of what I created.",
        "subreddit": "webdev",
        "score": 250,
        "num_comments": 60,
        "created_utc": 1234569000,
        "url": "https://reddit.com/r/webdev/sample_5",
        "permalink": "https://reddit.com/r/webdev/sample_5",
        "author": "newbie_dev",
        "comments": [
            {
                "id": "comment_9",
                "body": "Congratulations! Building your first full-stack app is a huge milestone. Keep building!",
                "score": 88,
                "created_utc": 1234569100,
                "author": "senior_dev"
            },
            {
                "id": "comment_10",
                "body": "Nice work! Consider adding TypeScript next - it will make your code more maintainable.",
                "score": 71,
                "created_utc": 1234569200,
                "author": "typescript_fan"
            }
        ]
    }
]

def create_sample_data():
    """Create sample data files for testing"""
    # Create raw data
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/reddit_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    print("Created sample raw data: data/raw/reddit_data.json")
    
    # Process the data
    from data.preprocess import RedditDataPreprocessor
    
    preprocessor = RedditDataPreprocessor()
    processed_data, stats = preprocessor.process_dataset(
        input_path="data/raw/reddit_data.json",
        output_path="data/processed/processed_data.json",
        min_length=100,
        max_length=2000,
        min_summary_length=20,
        max_summary_length=128
    )
    
    print(f"\nProcessed {stats['processed']} posts")
    print(f"Average post length: {stats['avg_post_length']:.1f} chars")
    print(f"Average summary length: {stats['avg_summary_length']:.1f} chars")
    
    # Create splits
    from data.dataset import split_data, save_splits
    
    train_data, val_data, test_data = split_data(
        processed_data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    save_splits(train_data, val_data, test_data, "data/splits")
    print(f"\nCreated splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print("\nSample data created successfully!")
    print("You can now run the training pipeline.")

if __name__ == "__main__":
    create_sample_data()

