# src/data_processing/create_mappings.py (Updated Version)

import pandas as pd
import pickle
from pathlib import Path
import argparse
import yaml
from collections import Counter

def generate_mappings_and_update_config(args):
    """
    Creates global ID maps for users, businesses, and categories,
    and updates the config file with their counts.
    """
    # --- Setup Paths ---
    input_file_reviews = Path(args.input_file_reviews)
    input_file_business = Path(args.input_file_business)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Process Users (from reviews) ---
    print(f"Processing users from {input_file_reviews}...")
    reviews_df = pd.read_json(input_file_reviews, lines=True)
    users_cols_to_keep = ['user_id']
    reviews_df = reviews_df[users_cols_to_keep]
    user_id_map = {id: i for i, id in enumerate(reviews_df['user_id'].unique())}
    n_users = len(user_id_map)
    
    # --- Process Businesses and Categories (from business.json) ---
    print(f"Processing businesses and categories from {input_file_business}...")
    business_df = pd.read_json(input_file_business, lines=True)
    business_cols_to_keep = ['business_id', 'categories']
    business_df = business_df[business_cols_to_keep]
    # Business Mapping
    business_id_map = {id: i for i, id in enumerate(business_df['business_id'].unique())}
    n_businesses = len(business_id_map)
    
    # Category Mapping
    # 1. Handle potential None values and split strings
    categories_series = business_df['categories'].dropna().str.split(', ')
    # 2. Flatten the list of lists into a single list of all category occurrences
    all_categories = [category for sublist in categories_series for category in sublist]
    # 3. Find all unique categories
    unique_categories = sorted(list(set(all_categories)))
    # 4. Create the map. We reserve index 0 for padding.
    category_map = {cat: i + 1 for i, cat in enumerate(unique_categories)}
    n_categories = len(category_map) + 1 # +1 for the padding value at index 0
    
    print(f"Found {n_users} unique users.")
    print(f"Found {n_businesses} unique businesses.")
    print(f"Found {len(unique_categories)} unique categories.")

    # --- Save Mappings ---
    with open(output_dir / "user_id_map.pkl", 'wb') as f: pickle.dump(user_id_map, f)
    with open(output_dir / "business_id_map.pkl", 'wb') as f: pickle.dump(business_id_map, f)
    with open(output_dir / "category_map.pkl", 'wb') as f: pickle.dump(category_map, f)
    print(f"Mappings saved to {output_dir}")

    # --- Update Config File ---
    print(f"Updating config file at {config_path}...")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    config['model']['n_users'] = n_users
    config['model']['n_businesses'] = n_businesses
    config['model']['n_categories'] = n_categories
    with open(config_path, 'w') as f: yaml.dump(config, f, sort_keys=False, indent=2)
    print("Config file updated successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_reviews", type=str, default="data/processed/restaurant_reviews.json")
    parser.add_argument("--input_file_business", type=str, default="data/unprocessed/yelp_academic_dataset_business.json") # We need the business file now
    parser.add_argument("--output_dir", type=str, default="saved_models")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    generate_mappings_and_update_config(args)