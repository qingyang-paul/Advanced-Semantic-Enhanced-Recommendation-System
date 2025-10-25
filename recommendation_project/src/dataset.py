# src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
# from .config import ROOT_DIR



class RecommendationDataset(Dataset):
    def __init__(self, reviews_path, users_path, businesses_path, user_id_map, business_id_map):
        print(f"正在从 {reviews_path} 加载数据...")
        
        reviews_df = pd.read_json(reviews_path, lines=True)
        users_df = pd.read_json(users_path, lines=True)
        businesses_df = pd.read_json(businesses_path, lines=True)

        user_features_df = users_df[['user_id', 'review_count', 'average_stars']]
        business_features_df = businesses_df[['business_id', 'stars', 'review_count']]
        
        merged_df = pd.merge(reviews_df, user_features_df, on='user_id')
        merged_df = pd.merge(merged_df, business_features_df, on='business_id', suffixes=('_user', '_business'))

        # --- Simplified Logic: Always use provided maps ---
        self.user_id_map = user_id_map
        self.business_id_map = business_id_map

        merged_df['user_idx'] = merged_df['user_id'].apply(lambda x: self.user_id_map.get(x, -1))
        merged_df['business_idx'] = merged_df['business_id'].apply(lambda x: self.business_id_map.get(x, -1))
        
        original_rows = len(merged_df)
        merged_df = merged_df[(merged_df['user_idx'] != -1) & (merged_df['business_idx'] != -1)]
        if len(merged_df) < original_rows:
            # This can happen if user.json/business.json has IDs not in the reviews file
            print(f"注意: 过滤掉了 {original_rows - len(merged_df)} 条包含未知ID的记录。")

        self.data = merged_df
        
        # The number of users/businesses is now defined by the global map, not the local data file
        self.n_users = len(self.user_id_map)
        self.n_businesses = len(self.business_id_map)

        print(f"数据加载完成。使用 {len(self.data)} 条评论。")
        print(f"模型将为 {self.n_users} 个全局用户和 {self.n_businesses} 个全局商户创建嵌入层。")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a single row of interaction data
        row = self.data.iloc[idx]

        # --- User Features ---
        user_features = {
            'id': torch.tensor(row['user_idx'], dtype=torch.long),
            'review_count': torch.tensor([row['review_count_user']], dtype=torch.float),
            'average_stars': torch.tensor([row['average_stars']], dtype=torch.float)
        }

        # --- Item Features ---
        item_features = {
            'id': torch.tensor(row['business_idx'], dtype=torch.long),
            'stars': torch.tensor([row['stars_business']], dtype=torch.float),
            'review_count': torch.tensor([row['review_count_business']], dtype=torch.float)
        }

        # --- Label (Target) ---
        label = torch.tensor(row['stars_user'], dtype=torch.float)

        # Return a dictionary matching the structure our Trainer expects
        return {
            'user': user_features,
            'item': item_features,
            'label': label
        }