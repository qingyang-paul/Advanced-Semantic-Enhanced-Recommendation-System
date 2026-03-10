# src/model/encoders.py (V6 - Specialized Encoders)

import torch
import torch.nn as nn

class UserEncoder(nn.Module):
    """
    专门用于编码用户特征的模块。
    它接收一系列连续的数值特征，并通过一个MLP来生成用户嵌入。
    """
    def __init__(self, n_continuous_features, hidden_layers_dims, output_dim):
        super().__init__()
        
        # 构建MLP，输入维度就是所有连续特征的数量
        layers = []
        current_dim = n_continuous_features
        for hidden_dim in hidden_layers_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) # 添加Dropout以增强泛化能力
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        # 将所有传入的连续特征张量拼接成一个向量
        # features是一个字典，其值为形状为 [batch_size, 1] 的张量
        continuous_vec = torch.cat(list(features.values()), dim=1)
        
        # 将拼接后的向量送入MLP
        return self.mlp(continuous_vec)

class BusinessEncoder(nn.Module):
    """
    专门用于编码商家特征的模块。
    它分别处理连续特征和类别特征，然后将它们融合在一起。
    """
    def __init__(self, n_continuous_features, 
                 n_categories, category_embedding_dim, 
                 hidden_layers_dims, output_dim):
        super().__init__()
        
        # 1. 类别特征的处理：EmbeddingBag
        self.category_embedding_bag = nn.EmbeddingBag(
            n_categories, category_embedding_dim, mode='mean', padding_idx=0
        )
        
        # 2. 构建MLP，其输入维度是连续特征和类别嵌入的总和
        mlp_input_dim = n_continuous_features + category_embedding_dim
        
        layers = []
        current_dim = mlp_input_dim
        for hidden_dim in hidden_layers_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        # 提取连续特征并拼接
        continuous_features_list = [v for k, v in features.items() if k != 'categories']
        continuous_vec = torch.cat(continuous_features_list, dim=1)
        
        # 处理类别特征
        category_vec = self.category_embedding_bag(features['categories'])
        
        # 将处理后的连续向量和类别向量拼接
        combined_vec = torch.cat([continuous_vec, category_vec], dim=1)
        
        # 将最终的特征向量送入MLP
        return self.mlp(combined_vec)