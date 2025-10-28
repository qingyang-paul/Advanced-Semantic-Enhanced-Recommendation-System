# src/train.py

import yaml
import torch
from torch.utils.data import DataLoader

# 假设这些是你接下来会写的文件
from dataset import RecommendationDataset # 我们稍后定义
from model.two_tower import TwoTowerModel # 我们稍后定义
from training.trainer import Trainer
import pickle

def main():
    # 1. 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 读取统一数据映射
    try:
        with open(config['paths']['user_id_map_path'], 'rb') as f:
            user_id_map = pickle.load(f)  # user_id_map 现在是一个字典
        with open(config['paths']['business_id_map_path'], 'rb') as f:
            business_id_map = pickle.load(f) # business_id_map 也是一个字典
    except FileNotFoundError:
        print("错误: 找不到ID映射文件。请先运行 create_mappings.py。")
        return

    # 设置设备
    # 依次判断 CUDA, MPS, CPU 是否可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device('cpu')
        print("No GPU available. Using CPU.")

    # 你可以打印出最终选择的设备
    print(f"使用设备: {device}")

    # 2. 加载数据
    dataset = RecommendationDataset(
        reviews_path=config['paths']['reviews_data_path'],
        users_path=config['paths']['users_data_path'],
        businesses_path=config['paths']['businesses_data_path'],
        user_id_map=user_id_map,
        business_id_map=business_id_map
    )

    # You can split it into train/val sets here if needed
    # For now, let's use the whole set for both for simplicity
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 3. 初始化模型
    # 模型的具体参数也从配置中读取
    model = TwoTowerModel(
        n_users=config['model']['n_users'],          # <-- 从配置读取
        n_businesses=config['model']['n_businesses'],  # <-- 从配置读取
        config=config['model']
    )

    # 4. 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.MSELoss() # 适用于二分类的损失函数



    # 5. 初始化训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )

    # 6. 开始训练
    trainer.train()

if __name__ == '__main__':
    main()