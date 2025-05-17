import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 导入各个模块
from data_fetcher import DataFetcher
from data_processor import DataProcessor
from model_data_processor import ModelDataProcessor
from models import SVMModel, RandomForestModel, MLPModel, RNNModel
from visualizer import ModelVisualizer
from file_manager import FileManager
# 从predict_eth.py导入预测功能
from predict_eth import predict_future, predict_multiple_timeframes

# 超参数配置字典
CONFIG = {
    # 数据获取参数
    'data': {
        'platforms': ['binance'],  # 交易平台列表
        'symbol': 'ETH-USDT',      # 交易对
        'timeframe': '5m',         # 改为 5m，这是币安支持的时间间隔
        'years_back': 3,           # 获取过去几年的数据
        'proxy': {'http': 'http://localhost:7897', 'https': 'http://localhost:7897'}
    },
    
    # 数据处理参数
    'data_processing': {
        'train_ratio': 0.7,        # 训练集比例
        'test_ratio': 0.2,         # 测试集比例
        'validation_ratio': 0.1,   # 验证集比例
        'features': ['open', 'high', 'low', 'close', 'volume'],  # 特征列表
        'target': 'close',         # 目标变量
        'time_steps': 30,          # 时间步长
        'smooth': False,           # 是否平滑数据
        'smooth_window': 5         # 平滑窗口大小
    },
    
    # 模型参数
    'models': {
        'svm': {
            'params': {'kernel': 'rbf', 'C': 1.0},
            'epochs': 100
        },
        'random_forest': {
            'params': {'n_estimators': 100, 'max_depth': 10},
            'epochs': 100
        },
        'mlp': {
            'params': {'hidden_layers': [64, 32], 'activation': 'relu'},
            'epochs': 100
        },
        'rnn': {
            'params': {'units': 50, 'layers': 2, 'dropout': 0.2},
            'epochs': 5
        }
    },
    
    # 训练参数
    'training': {
        'batch_size': 32,
        'early_stopping': True,
        'patience': 10
    },
    
    # 预测参数
    'prediction': {
        'time_period': {'unit': 'minutes', 'value': 10},  # 预测未来10分钟
        'timeframes': ['1m']       # 改为币安支持的时间间隔
    }
}

class ModelTrainer:
    """
    模型训练器，用于训练各种机器学习和深度学习模型
    """
    
    def __init__(self, model, train_data, validation_data=None, epochs=100, 
                 batch_size=32, early_stopping=False, patience=10):
        """
        初始化模型训练器
        
        参数:
            model: 要训练的模型
            train_data: 训练数据，包含X和y
            validation_data: 验证数据，包含X和y
            epochs: 训练轮次
            batch_size: 批次大小
            early_stopping: 是否使用早停
            patience: 早停耐心值
        """
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        
        # 设置日志
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.info(f"模型训练器初始化完成，模型类型: {type(model).__name__}")
    
    def train(self):
        """
        训练模型
        
        返回:
            dict: 训练历史记录
        """
        self.logger.info(f"开始训练模型: {self.model.model_name}")
        
        # 准备训练数据
        X_train = self.train_data['X']
        y_train = self.train_data['y']
        
        # 准备验证数据（如果有）
        X_val = None
        y_val = None
        if self.validation_data is not None:
            X_val = self.validation_data['X']
            y_val = self.validation_data['y']
        
        # 训练模型
        history = self.model.train(
            X_train, 
            y_train, 
            X_val=X_val, 
            y_val=y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            patience=self.patience
        )
        
        self.logger.info(f"模型 {self.model.model_name} 训练完成")
        return history


class ModelEvaluator:
    """
    模型评估器，用于评估模型性能
    """
    
    def __init__(self, model, test_data):
        """
        初始化模型评估器
        
        参数:
            model: 要评估的模型
            test_data: 测试数据，包含X和y
        """
        self.model = model
        self.test_data = test_data
        
        # 设置日志
        self.logger = logging.getLogger('ModelEvaluator')
        self.logger.info(f"模型评估器初始化完成，模型类型: {type(model).__name__}")
    
    def evaluate(self):
        """
        评估模型性能
        
        返回:
            dict: 评估结果
        """
        self.logger.info(f"开始评估模型: {self.model.model_name}")
        
        # 准备测试数据
        X_test = self.test_data['X']
        y_test = self.test_data['y']
        
        # 模型预测
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        # 计算方向准确率（预测涨跌方向是否正确）
        direction_accuracy = np.mean(
            (np.diff(y_test) > 0) == (np.diff(y_pred) > 0)
        )
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'predictions': y_pred,
            'actual': y_test
        }
        
        self.logger.info(f"模型 {self.model.model_name} 评估完成")
        self.logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, 方向准确率: {direction_accuracy:.4f}")
        
        return results


def main():
    """主函数：执行完整的数据获取、处理、模型训练和评估流程"""
    # 1. 数据获取
    # 初始化文件管理器
    file_manager = FileManager(base_dir='./data')
    
    # 从配置中获取数据参数
    platforms = CONFIG['data']['platforms']
    symbol = CONFIG['data']['symbol']
    start_date = (datetime.now().year - CONFIG['data']['years_back'])
    end_date = datetime.now()
    timeframe = CONFIG['data']['timeframe']
    proxy = CONFIG['data']['proxy']
    
    # 构建缓存文件名
    cache_filename = f"{symbol.replace('-', '_')}_{start_date}_{end_date.strftime('%Y%m%d')}_{timeframe}.csv"
    
    # 检查缓存目录中是否已有数据
    cache_dir = os.path.join('./data', 'raw_data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_filepath):
        # 如果缓存文件存在，直接加载
        logging.info(f"从缓存加载数据: {cache_filepath}")
        raw_data = file_manager.load_raw_data(cache_filename, exchange='cache')
        # 将DataFrame转换为字典格式，与从网络获取的数据格式保持一致
        if isinstance(raw_data, pd.DataFrame):
            # 根据platform列拆分数据
            platforms_in_data = raw_data['platform'].unique()
            raw_data_dict = {}
            for platform in platforms_in_data:
                platform_df = raw_data[raw_data['platform'] == platform].copy()
                platform_df = platform_df.drop(columns=['platform'])  # 删除platform列
                raw_data_dict[platform] = platform_df
            raw_data = raw_data_dict
    else:
        # 如果缓存文件不存在，从网络获取
        logging.info(f"从网络获取数据: {symbol}")
        data_fetcher = DataFetcher(
            platforms=platforms,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            proxy=proxy
        )
        raw_data = data_fetcher.fetch_data()
        
        # 保存到缓存
        if raw_data:
            merged_data = data_fetcher.merge_platform_data(raw_data)
            file_manager.save_raw_data(merged_data, cache_filename, exchange='cache')
            logging.info(f"数据已保存到缓存: {cache_filepath}")
            
    # 2. 基础数据处理
    data_processor = DataProcessor()
    processed_df = data_processor.process_data(raw_data)
    
    # 3. 模型数据处理
    model_data_processor = ModelDataProcessor(
        data=processed_df,
        train_ratio=CONFIG['data_processing']['train_ratio'],
        test_ratio=CONFIG['data_processing']['test_ratio'],
        validation_ratio=CONFIG['data_processing']['validation_ratio'],
        features=CONFIG['data_processing']['features'],
        target=CONFIG['data_processing']['target'],
        time_steps=CONFIG['data_processing']['time_steps'],
        smooth=CONFIG['data_processing']['smooth'],
        smooth_window=CONFIG['data_processing']['smooth_window']
    )
    train_data, test_data, validation_data = model_data_processor.split_data()
    
    # 4. 模型定义
    models = {
        'svm': SVMModel(params=CONFIG['models']['svm']['params']),
        'random_forest': RandomForestModel(params=CONFIG['models']['random_forest']['params']),
        'mlp': MLPModel(params=CONFIG['models']['mlp']['params']),
        'rnn': RNNModel(params=CONFIG['models']['rnn']['params'])
    }
    
    # 5. 文件管理器初始化
    file_manager = FileManager(base_dir='./saved_models')
    
    # 6. 可视化器初始化
    visualizer = ModelVisualizer()
    
    # 7. 模型训练、测试和评估
    results = {}
    saved_model_paths = {}  # 存储保存的模型路径
    
    for model_name, model in models.items():
        print(f"训练模型: {model_name}")
        
        # 训练模型，使用针对特定模型的轮数
        trainer = ModelTrainer(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=CONFIG['models'][model_name]['epochs'],
            batch_size=CONFIG['training']['batch_size'],
            early_stopping=CONFIG['training']['early_stopping'],
            patience=CONFIG['training']['patience']
        )
        training_history = trainer.train()
        
        # 保存模型并获取保存的路径
        model_path = file_manager.save_model(model, f"{model_name}_model")
        saved_model_paths[model_name] = model_path
        
        # 评估模型
        evaluator = ModelEvaluator(model=model, test_data=test_data)
        evaluation_results = evaluator.evaluate()
        results[model_name] = evaluation_results
        
        # 可视化训练过程
        visualizer.plot_training_history(training_history, model_name, show=False)
        
        # 获取模型预测值
        y_pred = model.predict(validation_data['X'])
        
        # 转换回原始尺度（反归一化）
        y_pred_original = model_data_processor.inverse_transform_y(y_pred)
        y_true_original = model_data_processor.inverse_transform_y(validation_data['y'])
        
        # 可视化预测结果（使用原始尺度数据）
        visualizer.plot_predictions(
            dates=validation_data['dates'],
            y_true=y_true_original,
            y_pred=y_pred_original,
            model_name=model_name,
            show= False
        )
    
    # 同样，对结果字典中的数据进行转换
    for model_name in results:
        results[model_name]['predictions'] = model_data_processor.inverse_transform_y(results[model_name]['predictions'])
        results[model_name]['actual'] = model_data_processor.inverse_transform_y(results[model_name]['actual'])
    
    # 8. 比较不同模型的性能（使用转换后的结果）
    visualizer.compare_models(results, dates=test_data['dates'], show=False)
    
    # 9. 保存处理后的数据
    file_manager.save_raw_data(processed_df, 'processed_eth_data.csv')
    
    # 10. 预测未来价格
    for model_name, model_path in saved_model_paths.items():
        predict_multiple_timeframes(
            model_path,
            time_period=CONFIG['prediction']['time_period'],
            symbol=CONFIG['data']['symbol'],
            timeframes=CONFIG['prediction']['timeframes']
        )
    
    print("ETH趋势预测程序执行完成！")


if __name__ == "__main__":
    main()
