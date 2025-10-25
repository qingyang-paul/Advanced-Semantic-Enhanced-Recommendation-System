# src/model/encoders.py

import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """
    一个通用的特征编码器，用于处理用户或物品的特征。
    它包含一个ID嵌入层和处理连续特征的全连接网络。
    """
    def __init__(self, n_unique_ids, id_embedding_dim, n_continuous_features, output_dim, hidden_layers_dims):
        """
        Args:
            n_unique_ids (int): 唯一ID的数量 (e.g., 用户数或商户数)。
            id_embedding_dim (int): ID嵌入向量的维度。
            n_continuous_features (int): 连续数值特征的数量。
            output_dim (int): 编码器最终输出的向量维度。
            hidden_layers_dims (list of int): MLP隐藏层的维度列表。
        """
        super().__init__()
        
        # 1. ID嵌入层
        self.id_embedding = nn.Embedding(n_unique_ids, id_embedding_dim)
        
        # 2. 用于融合所有特征的MLP
        # MLP的输入维度 = ID嵌入维度 + 连续特征数
        mlp_input_dim = id_embedding_dim + n_continuous_features
        
        layers = []
        current_dim = mlp_input_dim
        for hidden_dim in hidden_layers_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        """
        前向传播。
        
        Args:
            features (dict): 从Dataset传来的特征字典。
                             必须包含 'id' 和其他连续特征。
        
        Returns:
            torch.Tensor: 输出的编码向量。
        """
        # 提取ID嵌入向量
        id_vec = self.id_embedding(features['id'])
        
        # 提取并拼接所有连续特征
        # 注意: 我们假设连续特征已经是 (batch_size, 1) 的形状
        continuous_features_list = []
        for key, value in features.items():
            if key != 'id':
                continuous_features_list.append(value)
        
        continuous_vec = torch.cat(continuous_features_list, dim=1)
        
        # 将ID嵌入向量和连续特征向量拼接
        combined_vec = torch.cat([id_vec, continuous_vec], dim=1)
        
        # 通过MLP进行融合
        output = self.mlp(combined_vec)
        
        return output