# src/model/two_tower.py

import torch
import torch.nn as nn
from .encoders import FeatureEncoder # 从同目录下的encoders.py导入

class TwoTowerModel(nn.Module):
    """
    双塔推荐模型。
    """
    def __init__(self, n_users, n_businesses, config):
        """
        Args:
            n_users (int): 用户总数。
            n_businesses (int): 商户总数。
            config (dict): 包含模型超参数的配置字典。
        """
        super().__init__()
        
        user_config = config['user_tower']
        item_config = config['item_tower']
        
        # 1. 初始化用户塔
        self.user_encoder = FeatureEncoder(
            n_unique_ids=n_users,
            id_embedding_dim=user_config['id_embedding_dim'],
            n_continuous_features=user_config['n_continuous_features'],
            output_dim=config['output_dim'], # 两个塔的输出维度必须一致
            hidden_layers_dims=user_config['hidden_layers']
        )
        
        # 2. 初始化物品（商户）塔
        self.item_encoder = FeatureEncoder(
            n_unique_ids=n_businesses,
            id_embedding_dim=item_config['id_embedding_dim'],
            n_continuous_features=item_config['n_continuous_features'],
            output_dim=config['output_dim'],
            hidden_layers_dims=item_config['hidden_layers']
        )

    def forward(self, user_features, item_features):
        """
        前向传播。
        
        Args:
            user_features (dict): 用户的特征字典。
            item_features (dict): 物品的特征字典。
            
        Returns:
            torch.Tensor: 预测的评分（一个标量）。
        """
        # 1. 编码用户和物品特征，得到它们的向量表示
        user_embedding = self.user_encoder(user_features)
        item_embedding = self.item_encoder(item_features)
        
        # 2. 计算用户和物品向量的点积作为预测分
        # (batch_size, embed_dim) * (batch_size, embed_dim) -> (batch_size, embed_dim)
        # torch.sum(..., dim=1) -> (batch_size,)
        prediction = torch.sum(user_embedding * item_embedding, dim=1)
        
        return prediction