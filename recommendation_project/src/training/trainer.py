# src/training/trainer.py

import torch
from tqdm import tqdm  # 用于显示进度条

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, config):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型 (e.g., TwoTowerModel)
            optimizer: 优化器 (e.g., Adam)
            loss_fn: 损失函数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 'cuda' or 'cpu'
            config: 包含超参数的配置对象
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')

    def _train_epoch(self):
        """执行一个 epoch 的训练"""
        self.model.train()  # 设置为训练模式
        total_loss = 0
        
        # 使用tqdm来显示进度
        for batch in tqdm(self.train_loader, desc="Training"):
            # 将数据移动到指定设备
            # batch 的具体结构取决于你的 Dataset 如何实现
            user_features = {k: v.to(self.device) for k, v in batch['user'].items()}
            item_features = {k: v.to(self.device) for k, v in batch['item'].items()}
            labels = batch['label'].to(self.device)

            # 1. 清零梯度
            self.optimizer.zero_grad()
            
            # 2. 前向传播
            predictions = self.model(user_features, item_features)
            
            # 3. 计算损失
            loss = self.loss_fn(predictions, labels.float())
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新权重
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """执行一个 epoch 的验证"""
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        with torch.no_grad(): # 在此模式下，不计算梯度，节省计算资源
            for batch in tqdm(self.val_loader, desc="Validation"):
                user_features = {k: v.to(self.device) for k, v in batch['user'].items()}
                item_features = {k: v.to(self.device) for k, v in batch['item'].items()}
                labels = batch['label'].to(self.device)
                
                predictions = self.model(user_features, item_features)
                loss = self.loss_fn(predictions, labels.float())
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def train(self):
        """完整的训练流程"""
        print("开始训练...")
        for epoch in range(self.config['training']['epochs']):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 保存表现最好的模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("发现更好的模型，正在保存...")
                torch.save(self.model.state_dict(), self.config['paths']['best_model_path'])
        
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")