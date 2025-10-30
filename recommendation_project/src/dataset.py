# src/dataset.py (V4 - With Rich User Features)

import torch
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime

class RecommendationDataset(Dataset):
    def __init__(self, reviews_path, users_path, businesses_path, category_map, max_categories, is_train=False):
        print(f"Initializing feature-based dataset from: {reviews_path}")

        reviews_df = pd.read_json(reviews_path, lines=True)
        users_df = pd.read_json(users_path, lines=True)
        businesses_df = pd.read_json(businesses_path, lines=True)


        # --- 新增：加载主题画像文件 ---
        print("加载主题画像特征...")
        user_themes_df = pd.read_json("data/processed/user_theme_profiles.json", orient='index')
        business_themes_df = pd.read_json("data/processed/business_theme_profiles.json", orient='index')
        
        # --- 合并画像到主特征DataFrame ---
        # 使用 left merge，对于没有画像的实体，其特征值将为NaN
        users_df = pd.merge(users_df, user_themes_df, left_on='user_id', right_index=True, how='left')
        businesses_df = pd.merge(businesses_df, business_themes_df, left_on='business_id', right_index=True, how='left')
        
        # 获取主题列的名称，并用0填充缺失值
        self.theme_cols = list(user_themes_df.columns)
        users_df[self.theme_cols] = users_df[self.theme_cols].fillna(0)
        businesses_df[self.theme_cols] = businesses_df[self.theme_cols].fillna(0)

        # --- 1. Engineer New User Features ---
        print("Engineering new user features...")
        # Convert 'yelping_since' to datetime
        users_df['yelping_since'] = pd.to_datetime(users_df['yelping_since'])
        # Calculate account age in days from a fixed recent date (for consistency)
        users_df['account_age_days'] = (datetime(2025, 1, 1) - users_df['yelping_since']).dt.days

        # Count friends and elite years. Handle empty/None values gracefully.
        users_df['friend_count'] = users_df['friends'].apply(lambda x: len(x.split(',')) if x != 'None' and x else 0)
        users_df['elite_years_count'] = users_df['elite'].apply(lambda x: len(x.split(',')) if x else 0)

        # Sum up all compliments
        compliment_cols = [col for col in users_df.columns if col.startswith('compliment_')]
        users_df['total_compliments'] = users_df[compliment_cols].sum(axis=1)

        # Select all user feature columns we'll use
        user_feature_cols = [
            'user_id', 'review_count', 'average_stars', 'account_age_days', 
            'friend_count', 'elite_years_count', 'useful', 'funny', 'cool', 
            'fans', 'total_compliments'
        ]
        user_features_df = users_df[user_feature_cols]

        # --- 2. Select Business Features (no changes here) ---
        business_features_df = businesses_df[['business_id', 'stars', 'review_count', 'categories']]
        
        # --- 3. Merge DataFrames ---
        merged_df = pd.merge(reviews_df, user_features_df, on='user_id', suffixes=('_review', '_user'))
        merged_df = pd.merge(merged_df, business_features_df, on='business_id', suffixes=('_user', '_business'))


        # --- 4. Normalize Numerical User Features ---
        # For simplicity, we'll do basic min-max normalization here.
        # In a production system, you'd fit a scaler on the training set and save it.

        
        print("Normalizing numerical features...")
        self.numerical_user_cols = [col for col in user_feature_cols if col != 'user_id']

        # 检查哪些列被添加了后缀，并更新列表
        # 这样做更具扩展性，因为你不需要手动写死 'review_count_user'
        cols_to_normalize_updated = []
        for col in self.numerical_user_cols:
            col_with_suffix = col + '_user'
            # 如果原始列名不存在，但带后缀的列名存在，就用带后缀的
            if col not in merged_df.columns and col_with_suffix in merged_df.columns:
                cols_to_normalize_updated.append(col_with_suffix)
            else:
                # 否则，使用原始列名（例如 'fans', 'cool' 等没有冲突的列）
                cols_to_normalize_updated.append(col)
        self.numerical_user_cols = cols_to_normalize_updated

        self.data = merged_df
        self.category_map = category_map
        self.max_categories = max_categories
        
        print("Dataset initialized successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- User Features (Now with all normalized numerical features) ---
        user_features = {
            col: torch.tensor([row[col]], dtype=torch.float) for col in self.numerical_user_cols
        }

        # 新增：添加主题特征
        for col in self.theme_cols:
            user_features[f"theme_{col}"] = torch.tensor([row[f"{col}_user"]], dtype=torch.float) # 注意后缀 _user


        # --- Item Features (No changes here) ---
        category_indices = [0] * self.max_categories
        # ... (category processing logic is the same)
        if isinstance(row['categories'], str):
            categories = [cat.strip() for cat in row['categories'].split(',')]
            indices = [self.category_map.get(cat) for cat in categories if self.category_map.get(cat) is not None]
            for i in range(min(len(indices), self.max_categories)):
                category_indices[i] = indices[i]

        item_features = {
            'stars': torch.tensor([row['stars_business']], dtype=torch.float),
            'review_count': torch.tensor([row['review_count_business']], dtype=torch.float),
            'categories': torch.tensor(category_indices, dtype=torch.long)
        }

        # 新增：添加主题特征
        for col in self.theme_cols:
            item_features[f"theme_{col}"] = torch.tensor([row[f"{col}_business"]], dtype=torch.float) # 注意后缀 _business
            
        label = torch.tensor(row['stars_user'], dtype=torch.float)

        return {'user': user_features, 'item': item_features, 'label': label}