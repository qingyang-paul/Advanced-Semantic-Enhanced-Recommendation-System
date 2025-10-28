# src/model/two_tower.py (V7 - Concat+MLP Interaction)

import torch
import torch.nn as nn
from .encoders import UserEncoder, BusinessEncoder

class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        user_config = config['user_tower']
        item_config = config['item_tower']
        
        self.user_encoder = UserEncoder(
            n_continuous_features=user_config['n_continuous_features'],
            hidden_layers_dims=user_config['hidden_layers'],
            output_dim=config['output_dim']
        )
        
        self.item_encoder = BusinessEncoder(
            n_continuous_features=item_config['n_continuous_features'],
            n_categories=config['n_categories'],
            category_embedding_dim=item_config['category_embedding_dim'],
            hidden_layers_dims=item_config['hidden_layers'],
            output_dim=config['output_dim']
        )
        
        # --- 新增：定义交互层MLP ---
        interaction_config = config['interaction_mlp']
        # 输入维度是用户向量和商家向量拼接后的总维度
        mlp_input_dim = config['output_dim'] * 2
        
        layers = []
        current_dim = mlp_input_dim
        for hidden_dim in interaction_config['hidden_layers']:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(interaction_config.get('dropout', 0.3)))
            current_dim = hidden_dim
        
        # 最终输出层，输出一个值（预测的评分）
        layers.append(nn.Linear(current_dim, 1))
        
        self.interaction_mlp = nn.Sequential(*layers)


    def forward(self, user_features, item_features):
        # 1. 编码用户和商家特征（这部分不变）
        user_embedding = self.user_encoder(user_features)
        item_embedding = self.item_encoder(item_features)
        
        # --- 2. 交互方式改变 ---
        # 旧的点积方式:
        # prediction = torch.sum(user_embedding * item_embedding, dim=1)
        
        # 新的拼接+MLP方式:
        # 2.1 沿特征维度（dim=1）拼接两个向量
        combined_embedding = torch.cat([user_embedding, item_embedding], dim=1)
        
        # 2.2 将拼接后的向量送入交互MLP
        raw_prediction = self.interaction_mlp(combined_embedding)
        
        # 2.3 MLP输出的形状是 [batch_size, 1]，需要将其变为 [batch_size] 以匹配标签
        prediction = raw_prediction.squeeze(-1)
        
        return prediction