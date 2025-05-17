import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Union, Optional, Any
import pickle
from datetime import datetime
import xgboost as xgb
XGBOOST_AVAILABLE = True


class BaseModel:
    """
    模型基类，定义了所有模型共有的方法和属性
    """
    
    def __init__(self, model_name: str = "base_model"):
        """
        初始化模型基类
        
        参数:
            model_name (str): 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.history = {"train_loss": [], "val_loss": []}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f'Model-{model_name}')
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict:
        """
        训练模型的抽象方法
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史记录
        """
        raise NotImplementedError("子类必须实现train方法")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测的抽象方法
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        raise NotImplementedError("子类必须实现predict方法")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            X (np.ndarray): 特征
            y (np.ndarray): 真实标签
            
        返回:
            Dict[str, float]: 包含各种评估指标的字典
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法评估")
            return {}
        
        y_pred = self.predict(X)
        
        # 计算各种评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        self.logger.info(f"模型评估结果: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        参数:
            filepath (str): 保存路径
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法保存")
            return
        
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.logger.info(f"模型已保存至 {filepath}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        
        参数:
            filepath (str): 模型路径
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            self.logger.info(f"模型已从 {filepath} 加载")
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")


class MLModel(BaseModel):
    """
    机器学习模型类，包含SVM、随机森林、MLP等
    """
    
    def __init__(self, model_type: str = "rf", **kwargs):
        """
        初始化机器学习模型
        
        参数:
            model_type (str): 模型类型，可选 'svm', 'rf', 'mlp'
            **kwargs: 模型参数
        """
        super().__init__(model_name=model_type)
        self.model_type = model_type
        
        # 根据模型类型初始化相应的模型
        if model_type == "svm":
            self.model = SVR(**kwargs)
        elif model_type == "rf":
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == "mlp":
            self.model = MLPRegressor(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.logger.info(f"初始化 {model_type} 模型")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """
        训练机器学习模型
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证标签
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史记录
        """
        self.logger.info(f"开始训练 {self.model_type} 模型...")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 计算训练集损失
        y_train_pred = self.model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        self.history["train_loss"].append(train_loss)
        
        # 如果提供了验证集，计算验证集损失
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_loss = mean_squared_error(y_val, y_val_pred)
            self.history["val_loss"].append(val_loss)
            self.logger.info(f"训练完成: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}")
        else:
            self.logger.info(f"训练完成: 训练损失={train_loss:.4f}")
        
        self.is_trained = True
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法预测")
            return np.array([])
        
        return self.model.predict(X)


class DeepLearningModel(BaseModel):
    """
    深度学习模型基类，定义了PyTorch深度学习模型的通用方法
    """
    
    def __init__(self, model_name: str = "dl_model", device: str = None):
        """
        初始化深度学习模型
        
        参数:
            model_name (str): 模型名称
            device (str): 设备类型，'cuda' 或 'cpu'
        """
        super().__init__(model_name=model_name)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 优化器和损失函数
        self.optimizer = None
        self.criterion = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, batch_size: int = 32, epochs: int = 100, 
              learning_rate: float = 0.001, **kwargs) -> Dict:
        """
        训练深度学习模型
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            X_val (np.ndarray, optional): 验证特征
            y_val (np.ndarray, optional): 验证标签
            batch_size (int): 批次大小
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史记录
        """
        if self.model is None:
            self.logger.error("模型未初始化")
            return self.history
        
        # 将模型移至指定设备
        self.model = self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 转换数据为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 如果提供了验证集，也转换为张量
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        self.logger.info(f"开始训练，共 {epochs} 轮...")
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.criterion(outputs, targets.view_as(outputs))
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # 如果提供了验证集，计算验证损失
            if X_val is not None and y_val is not None:
                # 评估模式
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets.view_as(outputs))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.history["val_loss"].append(val_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}")
        
        self.is_trained = True
        self.logger.info("训练完成")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法预测")
            return np.array([])
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # 转换回NumPy数组
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str) -> None:
        """
        保存深度学习模型
        
        参数:
            filepath (str): 保存路径
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法保存")
            return
        
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存模型状态字典
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'history': self.history,
                'model_name': self.model_name
            }, filepath)
            
            self.logger.info(f"模型已保存至 {filepath}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载深度学习模型
        
        参数:
            filepath (str): 模型路径
        """
        try:
            # 加载模型
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果有优化器状态，也加载
            if checkpoint['optimizer_state_dict'] and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练历史
            self.history = checkpoint['history']
            
            self.is_trained = True
            self.logger.info(f"模型已从 {filepath} 加载")
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")


class MLPModel(DeepLearningModel):
    """
    多层感知机模型
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], output_dim: int = 1, 
                 dropout: float = 0.2, device: str = None):
        """
        初始化MLP模型
        
        参数:
            input_dim (int): 输入维度
            hidden_dims (List[int]): 隐藏层维度列表
            output_dim (int): 输出维度
            dropout (float): Dropout比率
            device (str): 设备类型
        """
        super().__init__(model_name="MLP", device=device)
        
        # 定义MLP模型
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 创建模型
        self.model = nn.Sequential(*layers)
        
        self.logger.info(f"初始化MLP模型: 输入维度={input_dim}, 隐藏层={hidden_dims}, 输出维度={output_dim}")


class RNNModel(DeepLearningModel):
    """
    循环神经网络模型
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2, rnn_type: str = "lstm", 
                 bidirectional: bool = False, device: str = None):
        """
        初始化RNN模型
        
        参数:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): RNN层数
            output_dim (int): 输出维度
            dropout (float): Dropout比率
            rnn_type (str): RNN类型，'lstm' 或 'gru'
            bidirectional (bool): 是否使用双向RNN
            device (str): 设备类型
        """
        super().__init__(model_name=f"{rnn_type.upper()}", device=device)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # 选择RNN类型
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 输出层
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            output_dim
        )
        
        # 创建模型
        self.model = self
        
        self.logger.info(
            f"初始化{rnn_type.upper()}模型: 输入维度={input_dim}, 隐藏层维度={hidden_dim}, "
            f"层数={num_layers}, 输出维度={output_dim}, 双向={bidirectional}"
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, output_dim]
        """
        # RNN层
        output, _ = self.rnn(x)
        
        # 我们只需要最后一个时间步的输出
        output = output[:, -1, :]
        
        # 全连接层
        output = self.fc(output)
        
        return output



class SVMModel(BaseModel):
    """
    支持向量机回归模型
    """
    
    def __init__(self, params: Dict = None):
        """
        初始化SVM模型
        
        参数:
            params (Dict): 模型参数，例如 {'kernel': 'rbf', 'C': 1.0}
        """
        super().__init__(model_name="SVM")
        
        # 设置默认参数
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
        
        # 更新参数
        if params:
            default_params.update(params)
        
        # 创建模型
        self.model = SVR(**default_params)
        self.logger.info(f"初始化SVM模型: {default_params}")
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"PyTorch使用设备: {self.device}")
    
    def _flatten_3d_data(self, X: np.ndarray) -> np.ndarray:
        """
        将3D数据转换为2D数据，适合SVM模型使用
        
        参数:
            X (np.ndarray): 3D输入数据，形状为 [n_samples, time_steps, n_features]
            
        返回:
            np.ndarray: 2D数据，形状为 [n_samples, time_steps * n_features]
        """
        if X.ndim == 3:
            self.logger.info(f"将3D数据 {X.shape} 转换为2D数据")
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, **kwargs) -> Dict:
        """
        训练SVM模型
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            X_val (np.ndarray): 验证特征
            y_val (np.ndarray): 验证标签
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史
        """
        self.logger.info(f"开始训练SVM模型，训练样本数: {X_train.shape[0]}")
        
        # 将3D数据转换为2D数据
        X_train_2d = self._flatten_3d_data(X_train)
        
        # 训练模型
        self.model.fit(X_train_2d, y_train)
        train_pred = self.model.predict(X_train_2d)
        
        # 计算训练损失
        train_loss = mean_squared_error(y_train, train_pred)
        self.history['train_loss'].append(train_loss)
        
        # 如果有验证数据，计算验证损失
        if X_val is not None and y_val is not None:
            X_val_2d = self._flatten_3d_data(X_val)
            val_pred = self.model.predict(X_val_2d)
            val_loss = mean_squared_error(y_val, val_pred)
            self.history['val_loss'].append(val_loss)
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        else:
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}")
        
        self.is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(X.shape[0])
        
        # 将3D数据转换为2D数据
        X_2d = self._flatten_3d_data(X)
        
        return self.model.predict(X_2d)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        joblib.dump(self.model, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path (str): 模型路径
        """
        self.model = joblib.load(path)
        self.is_trained = True
        self.logger.info(f"模型已从 {path} 加载")


class RandomForestModel(BaseModel):
    """
    随机森林回归模型，支持XGBoost GPU加速
    """
    
    def __init__(self, params: Dict = None, use_gpu: bool = True, gpu_id: int = 0):
        """
        初始化随机森林模型
        
        参数:
            params (Dict): 模型参数，例如 {'n_estimators': 100, 'max_depth': 10}
            use_gpu (bool): 是否使用GPU加速
            gpu_id (int): GPU设备ID
        """
        super().__init__(model_name="RandomForest")
        
        # 设置默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # 更新参数
        if params:
            default_params.update(params)
        
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # 尝试导入XGBoost
        try:
            import xgboost as xgb
            self.xgb_available = True
        except ImportError:
            self.xgb_available = False
            self.logger.warning("XGBoost未安装，将使用scikit-learn的RandomForest")
        
        # 创建模型
        if self.use_gpu and self.xgb_available:
            try:
                # 使用XGBoost的GPU加速版本
                self.use_xgb = True
                self.xgb_params = {
                    'objective': 'reg:squarederror',
                    'tree_method': 'hist',  # 使用hist替代gpu_hist
                    'device': 'cuda',       # 使用device替代gpu_id
                    'max_depth': default_params['max_depth'] if default_params['max_depth'] is not None else 6,
                    'eta': 0.1,
                    'random_state': default_params['random_state']
                }
                self.num_boost_round = default_params['n_estimators']  # 使用num_boost_round而非n_estimators
                self.model = None  # 将在训练时创建
                self.logger.info(f"初始化GPU加速XGBoost模型: {self.xgb_params}")
            except Exception as e:
                self.logger.warning(f"无法初始化GPU加速XGBoost模型: {str(e)}，回退到CPU版本")
                self.use_xgb = False
                self.model = RandomForestRegressor(**default_params)
                self.logger.info(f"初始化CPU版随机森林模型: {default_params}")
        else:
            self.use_xgb = False
            self.model = RandomForestRegressor(**default_params)
            self.logger.info(f"初始化CPU版随机森林模型: {default_params}")
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"PyTorch使用设备: {self.device}")
    
    def _flatten_3d_data(self, X: np.ndarray) -> np.ndarray:
        """
        将3D数据转换为2D数据，适合随机森林模型使用
        
        参数:
            X (np.ndarray): 3D输入数据，形状为 [n_samples, time_steps, n_features]
            
        返回:
            np.ndarray: 2D数据，形状为 [n_samples, time_steps * n_features]
        """
        if X.ndim == 3:
            self.logger.info(f"将3D数据 {X.shape} 转换为2D数据")
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, **kwargs) -> Dict:
        """
        训练随机森林模型
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            X_val (np.ndarray): 验证特征
            y_val (np.ndarray): 验证标签
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史
        """
        self.logger.info(f"开始训练随机森林模型，训练样本数: {X_train.shape[0]}")
        
        # 将3D数据转换为2D数据
        X_train_2d = self._flatten_3d_data(X_train)
        
        if self.use_xgb:
            try:
                import xgboost as xgb
                # 创建DMatrix数据格式
                dtrain = xgb.DMatrix(X_train_2d, label=y_train)
                
                # 训练模型
                self.model = xgb.train(self.xgb_params, dtrain, num_boost_round=self.num_boost_round)
                
                # 预测训练数据
                train_pred = self.model.predict(dtrain)
            except Exception as e:
                self.logger.warning(f"XGBoost GPU训练失败: {str(e)}，回退到CPU版本")
                self.use_xgb = False
                self.model = RandomForestRegressor(
                    n_estimators=self.num_boost_round,
                    max_depth=self.xgb_params['max_depth'],
                    random_state=self.xgb_params['random_state']
                )
                self.model.fit(X_train_2d, y_train)
                train_pred = self.model.predict(X_train_2d)
        else:
            # 使用CPU版本
            self.model.fit(X_train_2d, y_train)
            train_pred = self.model.predict(X_train_2d)
        
        # 计算训练损失
        train_loss = mean_squared_error(y_train, train_pred)
        self.history['train_loss'].append(train_loss)
        
        # 如果有验证数据，计算验证损失
        if X_val is not None and y_val is not None:
            X_val_2d = self._flatten_3d_data(X_val)
            
            if self.use_xgb:
                import xgboost as xgb
                dval = xgb.DMatrix(X_val_2d)
                val_pred = self.model.predict(dval)
            else:
                val_pred = self.model.predict(X_val_2d)
                
            val_loss = mean_squared_error(y_val, val_pred)
            self.history['val_loss'].append(val_loss)
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        else:
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}")
        
        self.is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(X.shape[0])
        
        # 将3D数据转换为2D数据
        X_2d = self._flatten_3d_data(X)
        
        if self.use_xgb:
            import xgboost as xgb
            dtest = xgb.DMatrix(X_2d)
            return self.model.predict(dtest)
        else:
            return self.model.predict(X_2d)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        if self.use_xgb:
            self.model.save_model(path)
        else:
            joblib.dump(self.model, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path (str): 模型路径
        """
        if self.use_xgb:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(path)
        else:
            self.model = joblib.load(path)
        self.is_trained = True
        self.logger.info(f"模型已从 {path} 加载")


class MLPModel(BaseModel):
    """
    多层感知机回归模型
    """
    
    def __init__(self, params: Dict = None):
        """
        初始化MLP模型
        
        参数:
            params (Dict): 模型参数，例如 {'hidden_layers': [64, 32], 'activation': 'relu'}
        """
        super().__init__(model_name="MLP")
        
        # 设置默认参数
        default_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'random_state': 42
        }
        
        # 更新参数
        if params:
            # 特殊处理hidden_layers参数
            if 'hidden_layers' in params:
                default_params['hidden_layer_sizes'] = tuple(params.pop('hidden_layers'))
            default_params.update(params)
        
        # 创建模型
        self.model = MLPRegressor(**default_params)
        self.logger.info(f"初始化MLP模型: {default_params}")
    
    def _flatten_3d_data(self, X: np.ndarray) -> np.ndarray:
        """
        将3D数据转换为2D数据，适合MLP模型使用
        
        参数:
            X (np.ndarray): 3D输入数据，形状为 [n_samples, time_steps, n_features]
            
        返回:
            np.ndarray: 2D数据，形状为 [n_samples, time_steps * n_features]
        """
        if X.ndim == 3:
            self.logger.info(f"将3D数据 {X.shape} 转换为2D数据")
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, **kwargs) -> Dict:
        """
        训练MLP模型
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练标签
            X_val (np.ndarray): 验证特征
            y_val (np.ndarray): 验证标签
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史
        """
        self.logger.info(f"开始训练MLP模型，训练样本数: {X_train.shape[0]}")
        
        # 将3D数据转换为2D数据
        X_train_2d = self._flatten_3d_data(X_train)
        
        # 训练模型
        self.model.fit(X_train_2d, y_train)
        
        # 计算训练损失
        train_pred = self.model.predict(X_train_2d)
        train_loss = mean_squared_error(y_train, train_pred)
        self.history['train_loss'] = self.model.loss_curve_
        
        # 如果有验证数据，计算验证损失
        if X_val is not None and y_val is not None:
            X_val_2d = self._flatten_3d_data(X_val)
            val_pred = self.model.predict(X_val_2d)
            val_loss = mean_squared_error(y_val, val_pred)
            self.history['val_loss'].append(val_loss)
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        else:
            self.logger.info(f"训练完成，训练损失: {train_loss:.4f}")
        
        self.is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(X.shape[0])
        
        # 将3D数据转换为2D数据
        X_2d = self._flatten_3d_data(X)
        
        return self.model.predict(X_2d)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        joblib.dump(self.model, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path (str): 模型路径
        """
        self.model = joblib.load(path)
        self.is_trained = True
        self.logger.info(f"模型已从 {path} 加载")

class RNNModel(BaseModel):
    """
    循环神经网络回归模型
    """
    
    def __init__(self, params: Dict = None):
        """
        初始化RNN模型
        
        参数:
            params (Dict): 模型参数，例如 {'units': 50, 'layers': 2, 'dropout': 0.2}
        """
        super().__init__(model_name="RNN")
        
        # 设置默认参数
        default_params = {
            'input_dim': 5,  # 默认特征数量
            'hidden_dim': 50,
            'output_dim': 1,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False,
            'rnn_type': 'lstm',
            'learning_rate': 0.001
        }
        
        # 更新参数
        if params:
            # 特殊处理units参数
            if 'units' in params:
                default_params['hidden_dim'] = params.pop('units')
            # 特殊处理layers参数
            if 'layers' in params:
                default_params['num_layers'] = params.pop('layers')
            default_params.update(params)
        
        # 保存参数
        self.params = default_params
        
        # 创建PyTorch模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 模型将在train方法中创建，因为我们需要知道输入维度
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        self.logger.info(f"初始化RNN模型: {default_params}")
    
    def _create_model(self, input_dim):
        """
        创建RNN模型
        
        参数:
            input_dim (int): 输入特征维度
        """
        # 更新输入维度
        self.params['input_dim'] = input_dim
        
        # 创建RNN模型
        self.model = RNNModule(
            input_dim=self.params['input_dim'],
            hidden_dim=self.params['hidden_dim'],
            output_dim=self.params['output_dim'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            bidirectional=self.params['bidirectional'],
            rnn_type=self.params['rnn_type']
        ).to(self.device)
        
        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        self.criterion = nn.MSELoss()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, epochs: int = 10, batch_size: int = 32, 
              early_stopping: bool = False, patience: int = 10, **kwargs) -> Dict:
        """
        训练RNN模型
        
        参数:
            X_train (np.ndarray): 训练特征，形状为 [n_samples, seq_len, n_features]
            y_train (np.ndarray): 训练标签，形状为 [n_samples]
            X_val (np.ndarray): 验证特征，形状为 [n_samples, seq_len, n_features]
            y_val (np.ndarray): 验证标签，形状为 [n_samples]
            epochs (int): 训练轮次
            batch_size (int): 批次大小
            early_stopping (bool): 是否使用早停
            patience (int): 早停耐心值
            **kwargs: 其他参数
            
        返回:
            Dict: 训练历史
        """
        self.logger.info(f"开始训练RNN模型，训练样本数: {X_train.shape[0]}")
        
        # 创建模型（如果尚未创建）
        if self.model is None:
            self._create_model(X_train.shape[2])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练历史
        self.history = {"train_loss": [], "val_loss": []}
        
        # 早停变量
        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.history['val_loss'].append(val_loss)
                
                # 早停检查
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                    
                    if no_improve_epochs >= patience:
                        self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                        break
                
                self.logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            else:
                self.logger.info(f"轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}")
        
        self.is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (np.ndarray): 预测特征，形状为 [n_samples, seq_len, n_features]
            
        返回:
            np.ndarray: 预测结果，形状为 [n_samples]
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(X.shape[0])
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.reshape(-1)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法保存")
            return
        
        # 保存模型参数和状态
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params,
            'history': self.history
        }
        torch.save(state, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path (str): 模型路径
        """
        # 加载模型状态
        state = torch.load(path, map_location=self.device)
        
        # 恢复参数
        self.params = state['params']
        
        # 创建模型
        self._create_model(self.params['input_dim'])
        
        # 加载模型参数
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # 恢复历史记录
        self.history = state['history']
        
        self.is_trained = True
        self.logger.info(f"模型已从 {path} 加载")


class RNNModule(nn.Module):
    """
    RNN模块，用于创建LSTM或GRU模型
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, bidirectional=False, rnn_type="lstm"):
        """
        初始化RNN模块
        
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            num_layers (int): RNN层数
            dropout (float): Dropout比率
            bidirectional (bool): 是否使用双向RNN
            rnn_type (str): RNN类型，'lstm'或'gru'
        """
        super(RNNModule, self).__init__()
        
        # 设置日志
        self.logger = logging.getLogger('RNNModule')
        
        # 选择RNN类型
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 计算RNN输出维度
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 全连接层
        self.fc = nn.Linear(rnn_output_dim, output_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入数据，形状为 [batch_size, seq_len, input_dim]
            
        返回:
            Tensor: 输出数据，形状为 [batch_size, output_dim]
        """
        # RNN前向传播
        rnn_out, _ = self.rnn(x)
        
        # 我们只需要最后一个时间步的输出
        last_out = rnn_out[:, -1, :]
        
        # 通过全连接层
        output = self.fc(last_out)
        
        return output
