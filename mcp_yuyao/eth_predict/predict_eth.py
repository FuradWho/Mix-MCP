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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    
    # 预测参数
    'prediction': {
        'time_period': {'unit': 'minutes', 'value': 60},  # 预测未来10分钟
        'timeframes': ['1m']       # 改为币安支持的时间间隔
    }
}

def predict_future(model_path, time_period=None, symbol='ETH-USDT', timeframe='5m'):
    """
    使用保存的模型预测未来一段时间的价格走势
    
    参数:
        model_path (str): 模型文件路径
        time_period (dict): 预测时间周期，格式为{'unit': 单位, 'value': 数值}
                           单位可以是'minutes', 'hours', 'days'
                           例如: {'unit': 'minutes', 'value': 10} 表示预测未来10分钟
                           如果为None，默认预测1天
        symbol (str): 交易对符号
        timeframe (str): 时间周期，支持'1m', '3m', '5m', '10m', '15m', '1h', '4h', '1d'
        
    返回:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    # 如果time_period为None，默认预测1天
    if time_period is None:
        time_period = {'unit': 'days', 'value': 1}
    
    unit = time_period['unit']
    value = time_period['value']
    
    logging.info(f"开始预测未来{value}{unit}的{symbol}价格走势，时间周期: {timeframe}")
    
    # ==================== 1. 模型加载 ====================
    # 从文件名中提取模型类型
    model_filename = os.path.basename(model_path)
    model_type = model_filename.split('_')[0]  # 提取模型类型（如svm, random, mlp等）
    
    # 初始化文件管理器并加载模型
    file_manager = FileManager(base_dir='./saved_models')
    
    # 根据模型类型创建相应的模型实例
    if model_type == 'svm':
        model = SVMModel()
    elif model_type == 'random':
        model = RandomForestModel()
    elif model_type == 'mlp':
        model = MLPModel()
    elif model_type == 'rnn':
        model = RNNModel()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型
    model = file_manager.load_model(model_path)
    logging.info(f"已加载模型: {model_path}")
    
    del file_manager
    
    # ==================== 2. 数据获取与处理 ====================
    platforms = CONFIG['data']['platforms']
    file_manager = FileManager(base_dir='./data')
    
    # 计算最小需要的数据量
    time_steps = CONFIG['data_processing']['time_steps']  # 模型使用的时间步数
    
    # 根据timeframe计算需要获取的最小天数
    timeframe_minutes = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '10m': 10,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    minutes_per_step = timeframe_minutes.get(timeframe, 1)
    
    # 计算需要的最小数据时间范围
    total_minutes_needed = time_steps * minutes_per_step
    days_needed = max(2, total_minutes_needed // (24 * 60) + 1)  # 至少获取2天的数据作为缓冲
    
    # 更新数据获取的时间范围
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days_needed)
    
    # 从网络获取数据
    logging.info(f"从网络获取数据: {symbol}, 时间范围: {start_date} 至 {end_date}")
    data_fetcher = DataFetcher(
        platforms=platforms,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        proxy=CONFIG['data']['proxy']
    )
    raw_data = data_fetcher.fetch_data()
            
    # 基础数据处理
    data_processor = DataProcessor()
    processed_df = data_processor.process_data(raw_data)
    
    # 模型数据处理
    model_data_processor = ModelDataProcessor(
        data=processed_df,
        train_ratio=0,
        test_ratio=1.0,
        validation_ratio=0,
        features=CONFIG['data_processing']['features'],
        target=CONFIG['data_processing']['target'],
        time_steps=CONFIG['data_processing']['time_steps'],
        smooth=CONFIG['data_processing']['smooth'],
        smooth_window=CONFIG['data_processing']['smooth_window']
    )
    
    # 获取最近的数据作为预测输入
    _, recent_data, _ = model_data_processor.split_data()

    # ==================== 3. 预测未来价格 ====================
    # 计算需要预测的时间步数
    timeframe_steps = {
        '1m': 60,  # 每小时60步
        '3m': 20,  # 每小时20步
        '5m': 12,  # 每小时12步
        '10m': 6,  # 每小时6步
        '15m': 4,  # 每小时4步
        '1h': 1,   # 每小时1步
        '4h': 0.25,  # 每小时0.25步
        '1d': 1/24  # 每小时1/24步
    }
    
    steps_per_hour = timeframe_steps.get(timeframe, 1)
    if timeframe not in timeframe_steps:
        logging.warning(f"未知的timeframe: {timeframe}，默认使用每小时1个时间步")
    
    # 根据时间单位计算预测步数
    if unit == 'minutes':
        prediction_steps = int((value / 60) * steps_per_hour)
    elif unit == 'hours':
        prediction_steps = int(value * steps_per_hour)
    elif unit == 'days':
        prediction_steps = int(value * 24 * steps_per_hour)
    else:
        raise ValueError(f"不支持的时间单位: {unit}，支持的单位有: minutes, hours, days")
    
    logging.info(f"将预测未来 {prediction_steps} 个时间步")
    
    # 初始化预测结果列表
    predictions = []
    current_input = recent_data['X'][-1:]  # 获取最近的一个输入样本
    
    # 逐步预测未来价格
    for i in range(prediction_steps):
        # 预测下一个时间步
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0])
        
        # 更新输入数据（移除最早的时间步，添加新预测的时间步）
        if model_type == 'rnn':
            # RNN模型的输入更新
            new_input = np.copy(current_input)
            new_input[0, :-1, :] = new_input[0, 1:, :]
            
            # 为所有特征创建合理的预测值
            last_values = new_input[0, -2, :]
            random_changes = np.random.normal(0, 0.001, size=last_values.shape)
            new_values = last_values + random_changes
            new_values[-1] = next_pred[0]  # 最后一个是预测的收盘价
            
            new_input[0, -1, :] = new_values
            current_input = new_input
        else:
            # 其他模型的输入更新
            new_input = np.copy(current_input)
            feature_size = new_input.shape[1] // model_data_processor.time_steps
            new_input[0, :-feature_size] = new_input[0, feature_size:]
            new_input[0, -1] = next_pred[0]  # 更新最后一个close价格
            current_input = new_input
    
    # ==================== 4. 结果处理与可视化 ====================
    # 转换预测结果回原始尺度
    predictions = np.array(predictions)
    predictions_original = model_data_processor.inverse_transform_y(predictions)
    
    # 创建预测结果DataFrame
    last_date = processed_df.index[-1]
    
    # 过滤历史数据，只保留最近一天的数据
    one_day_ago = processed_df.index[-1] - pd.Timedelta(days=1)
    historical_data_filtered = processed_df[processed_df.index >= one_day_ago]

    # timeframe到pandas频率转换
    timeframe_freq = {
        '1m': '1min',
        '3m': '3min',
        '5m': '5min',
        '10m': '10min',
        '15m': '15min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    freq = timeframe_freq.get(timeframe, '1D')
    if timeframe not in timeframe_freq:
        logging.warning(f"未知的timeframe: {timeframe}，默认使用1天频率")

    future_dates = pd.date_range(start=last_date, periods=prediction_steps+1, freq=freq)[1:]

    # 创建预测结果DataFrame
    prediction_df = pd.DataFrame({
        'predicted_close': predictions_original
    }, index=future_dates)
    
    # 可视化预测结果
    visualizer = ModelVisualizer()
    visualizer.plot_future_prediction(
        historical_data=historical_data_filtered,  # 仅使用过去一天的数据 
        prediction_df=prediction_df, 
        model_name=model_type, 
        save=True, 
        show=True
    )
    
    logging.info(f"预测完成，共预测了 {len(prediction_df)} 个时间点")
    return prediction_df


def predict_multiple_timeframes(model_path, time_period=None, symbol='ETH-USDT', timeframes=None):
    """
    使用保存的模型预测多个时间周期的未来价格走势
    
    参数:
        model_path (str): 模型文件路径
        time_period (dict): 预测时间周期，格式为{'unit': 单位, 'value': 数值}
                           单位可以是'minutes', 'hours', 'days'
                           例如: {'unit': 'minutes', 'value': 10} 表示预测未来10分钟
                           如果为None，默认预测1天
        symbol (str): 交易对符号
        timeframes (list): 要预测的时间周期列表，默认为['1m', '3m', '5m', '10m', '15m', '1h']
        
    返回:
        dict: 包含各个时间周期预测结果的字典
    """
    if timeframes is None:
        timeframes = ['1m']
    
    # 如果time_period为None，默认预测1天
    if time_period is None:
        time_period = {'unit': 'days', 'value': 1}
    
    logging.info(f"开始多时间周期预测: {timeframes}, 预测时间: {time_period['value']} {time_period['unit']}")
    
    results = {}
    for tf in timeframes:
        logging.info(f"预测时间周期: {tf}")
        prediction_df = predict_future(model_path, time_period, symbol, tf)
        results[tf] = prediction_df
        
        # 如果是10分钟后的预测，将结果写入文件
        if time_period['unit'] == 'minutes' and time_period['value'] == 10:
            # 获取10分钟后的预测值
            ten_min_prediction = prediction_df.iloc[0]['predicted_close']
            # 写入文件
            with open('predict-yuyao.txt', 'w') as f:
                f.write(f"{ten_min_prediction}")
            logging.info(f"已将10分钟后的预测值 {ten_min_prediction} 写入 predict-yuyao.txt 文件")
        
    return results


if __name__ == "__main__":
    # 使用默认配置进行预测
    model_path = r'D:\Project\hack-v1-tm24-pharos\eth-predict\saved_models\models\generic\random_forest_model_20250413_161219.pkl'
    
    predict_multiple_timeframes(
        model_path, 
        time_period=CONFIG['prediction']['time_period'],
        symbol=CONFIG['data']['symbol'], 
        timeframes=CONFIG['prediction']['timeframes']
    )
