# src/evaluate.py

import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import pickle


# 导入我们自己写的模块
from dataset import RecommendationDataset
from model.two_tower import TwoTowerModel

def evaluate_model():
    """
    加载训练好的模型并在测试集上进行评估。
    """
    # 1. 加载配置
    try:
        with open('../configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误: config.yaml 文件未找到。请确保你在项目的根目录下运行此脚本。")
        return

    # 读取统一数据映射
    try:
        with open(config['paths']['user_id_map_path'], 'rb') as f:
            user_id_map = pickle.load(f)  # user_id_map 现在是一个字典
        with open(config['paths']['business_id_map_path'], 'rb') as f:
            business_id_map = pickle.load(f) # business_id_map 也是一个字典
    except FileNotFoundError:
        print("错误: 找不到ID映射文件。请先运行 create_mappings.py。")
        return
    
    # 2. 设置设备
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


    print(f"使用设备: {device}")

    # 3. 加载测试数据
    # 假设你已经将数据分为了训练集和测试集
    # 这里我们加载测试集数据，需要确保配置文件中有 test_data_path

    # 2. 加载数据
    dataset = RecommendationDataset(
        reviews_path=config['paths']['reviews_data_path'],
        users_path=config['paths']['users_data_path'],
        businesses_path=config['paths']['businesses_data_path'],
        user_id_map=user_id_map,
        business_id_map=business_id_map
    )
    
    try:
        test_dataset = RecommendationDataset(
            reviews_path=config['paths']['test_reviews_path'],
            users_path=config['paths']['users_data_path'],
            businesses_path=config['paths']['businesses_data_path'],
            user_id_map=user_id_map,
            business_id_map=business_id_map
        )
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    except KeyError:
        print("错误: 配置文件中缺少 'test_reviews_path'。请添加测试集评论文件的路径。")
        return
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        return


    # 4. 初始化模型架构
    # 模型结构必须和训练时完全一致

    model = TwoTowerModel(
        n_users=dataset.n_users,
        n_businesses=dataset.n_businesses,
        config=config['model']
    )

    # 5. 加载训练好的模型权重
    model_path = config['paths']['best_model_path']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功从 {model_path} 加载模型权重。")
    except FileNotFoundError:
        print(f"错误: 找不到已保存的模型文件 at {model_path}。请先运行 train.py 进行训练。")
        return
    
    model.to(device)
    model.eval()  # *** 非常重要：将模型设置为评估模式 ***

    # 6. 开始评估
    all_predictions = []
    all_labels = []

    # with torch.no_grad() 可以禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 将数据移动到设备
            user_features = {k: v.to(device) for k, v in batch['user'].items()}
            item_features = {k: v.to(device) for k, v in batch['item'].items()}
            labels = batch['label'] # labels不需要移动到GPU

            # 模型预测
            predictions = model(user_features, item_features)

            # 将预测结果和真实标签从GPU移回CPU，并存入列表
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. 计算并打印评估指标
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_predictions)

    print("\n--- 评估结果 ---")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print("--------------------")
    print(f"提示: RMSE/MAE 的值越低，表示模型的预测越精准。")


if __name__ == '__main__':
    evaluate_model()