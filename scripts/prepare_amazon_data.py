"""
Script to preprocess Amazon Beauty reviews into sequential interaction format
"""
import json
import gzip
from collections import defaultdict
from datetime import datetime

def load_amazon_reviews(file_path):
    """
    Load Amazon reviews from .json.gz or .json file
    Expected format: one JSON object per line
    """
    reviews = []

    if file_path.endswith('.gz'):
        opener = gzip.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'

    print(f"Loading reviews from {file_path}...")
    with opener(file_path, mode, encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue

    print(f"Loaded {len(reviews)} reviews")
    return reviews

def build_user_sequences(reviews, min_seq_len=3):
    """
    Group reviews by user and create chronological sequences

    Returns:
        user_sequences: dict of {user_id: [(item_id, timestamp), ...]}
    """
    user_interactions = defaultdict(list)

    for review in reviews:
        user_id = review.get('reviewerID')
        item_id = review.get('asin')
        timestamp = review.get('unixReviewTime', 0)

        if user_id and item_id:
            user_interactions[user_id].append((item_id, timestamp))

    # Sort by timestamp and filter short sequences
    user_sequences = {}
    for user_id, interactions in user_interactions.items():
        # Sort chronologically
        sorted_interactions = sorted(interactions, key=lambda x: x[1])
        item_sequence = [item_id for item_id, _ in sorted_interactions]

        # Only keep users with enough interactions
        if len(item_sequence) >= min_seq_len:
            user_sequences[user_id] = item_sequence

    print(f"Built sequences for {len(user_sequences)} users (min_len={min_seq_len})")
    return user_sequences

def create_train_test_split(user_sequences, test_ratio=0.1):
    """
    Split users into train/test sets
    For each sequence, use last item as target, rest as history
    """
    from random import shuffle, seed

    seed(42)
    user_ids = list(user_sequences.keys())
    shuffle(user_ids)

    split_idx = int(len(user_ids) * (1 - test_ratio))
    train_users = user_ids[:split_idx]
    test_users = user_ids[split_idx:]

    train_data = []
    test_data = []

    for user_id in train_users:
        seq = user_sequences[user_id]
        # Create training samples: predict each item from history
        for i in range(1, len(seq)):
            train_data.append({
                'user_id': user_id,
                'history': seq[:i],
                'target': seq[i]
            })

    for user_id in test_users:
        seq = user_sequences[user_id]
        # For test: only use last item as target
        test_data.append({
            'user_id': user_id,
            'history': seq[:-1],
            'target': seq[-1]
        })

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_data, test_data

def get_item_stats(user_sequences):
    """Get statistics about items in the dataset"""
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)

    # Create item to index mapping
    item2idx = {item: idx+1 for idx, item in enumerate(sorted(all_items))}  # 0 reserved for padding

    print(f"Unique items: {len(all_items)}")
    print(f"Item vocab size (with padding): {len(item2idx) + 1}")

    return item2idx

def save_processed_data(train_data, test_data, item2idx, output_dir='data/beauty'):
    """Save processed data"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/train_sequences.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{output_dir}/test_sequences.json', 'w') as f:
        json.dump(test_data, f)

    with open(f'{output_dir}/item2idx.json', 'w') as f:
        json.dump(item2idx, f)

    print(f"\n✅ Saved processed data to {output_dir}/")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prepare_amazon_data.py <path_to_reviews.json.gz>")
        print("Example: python prepare_amazon_data.py data/beauty/reviews_Beauty_5.json.gz")
        sys.exit(1)

    review_file = sys.argv[1]

    # Load and process
    reviews = load_amazon_reviews(review_file)
    user_sequences = build_user_sequences(reviews, min_seq_len=3)
    item2idx = get_item_stats(user_sequences)
    train_data, test_data = create_train_test_split(user_sequences)
    save_processed_data(train_data, test_data, item2idx)

    print("\n📊 Dataset Statistics:")
    print(f"  Total users: {len(user_sequences)}")
    print(f"  Total items: {len(item2idx)}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"\nSample training instance:")
    print(f"  {train_data[0]}")
