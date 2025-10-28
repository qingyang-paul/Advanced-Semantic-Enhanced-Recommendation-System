# src/model/two_tower.py (V6 - Using Specialized Encoders)

import torch
import torch.nn as nn
# 导入新的Encoder类
from .encoders import UserEncoder, BusinessEncoder

class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        user_config = config['user_tower']
        item_config = config['item_tower']
        
        # 1. 初始化专用的UserEncoder
        self.user_encoder = UserEncoder(
            n_continuous_features=user_config['n_continuous_features'],
            hidden_layers_dims=user_config['hidden_layers'],
            output_dim=config['output_dim']
        )
        
        # 2. 初始化专用的BusinessEncoder
        self.item_encoder = BusinessEncoder(
            n_continuous_features=item_config['n_continuous_features'],
            n_categories=config['n_categories'],
            category_embedding_dim=item_config['category_embedding_dim'],
            hidden_layers_dims=item_config['hidden_layers'],
            output_dim=config['output_dim']
        )

    def forward(self, user_features, item_features):
        # 分别调用各自的encoder
        user_embedding = self.user_encoder(user_features)
        item_embedding = self.item_encoder(item_features)
        
        # 最终的交互方式保持不变
        prediction = torch.sum(user_embedding * item_embedding, dim=1)
        
        return prediction