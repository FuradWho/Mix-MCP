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
        'timeframes': ['5m']       # 改为币安支持的时间间隔
    }
}

def predict_future(model_path, time_period=None, symbol='ETH-USDT', timeframe='5m'):
    """
    Using saved model to predict future price trends for a period of time
    
    Parameters:
        model_path (str): Model file path
        time_period (dict): Prediction time period, format {'unit': unit, 'value': value}
                           Units can be 'minutes', 'hours', 'days'
                           Example: {'unit': 'minutes', 'value': 10} means predict 10 minutes in the future
                           If None, default predicts 1 day
        symbol (str): Trading pair symbol
        timeframe (str): Time frame, supports '1m', '3m', '5m', '10m', '15m', '1h', '4h', '1d'
        
    Returns:
        pd.DataFrame: DataFrame containing prediction results
    """
    # If time_period is None, default predict 1 day
    if time_period is None:
        time_period = {'unit': 'days', 'value': 1}
    
    unit = time_period['unit']
    value = time_period['value']
    
    logging.info(f"Starting prediction for {symbol} for the next {value} {unit}, timeframe: {timeframe}")
    
    # ==================== 1. Model Loading ====================
    # Extract model type from filename
    model_filename = os.path.basename(model_path)
    model_type = model_filename.split('_')[0]  # Extract model type (e.g., svm, random, mlp, etc.)
    
    # Initialize file manager and load model
    file_manager = FileManager(base_dir='./eth_predict/saved_models')
    
    # Create model instance based on model type
    if model_type == 'svm':
        model = SVMModel()
    elif model_type == 'random':
        model = RandomForestModel()
    elif model_type == 'mlp':
        model = MLPModel()
    elif model_type == 'rnn':
        model = RNNModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model
    model = file_manager.load_model(model_path)
    logging.info(f"Model loaded: {model_path}")
    
    # ==================== 2. Data Acquisition and Processing ====================
    platforms = CONFIG['data']['platforms']
    file_manager = FileManager(base_dir='./eth_predict/data')
    
    # Calculate minimum required data amount
    time_steps = CONFIG['data_processing']['time_steps']  # Number of time steps used by the model
    
    # Calculate required minimum days based on timeframe
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
    
    # Calculate minimum data time range needed
    total_minutes_needed = time_steps * minutes_per_step
    days_needed = max(2, total_minutes_needed // (24 * 60) + 1)  # Get at least 2 days of data as buffer
    
    # Update data acquisition time range
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days_needed)
    
    # Try to load data from cache first
    cache_dir = os.path.join('./eth_predict/data', 'raw_data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if there's any cached data file
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith(f"{symbol.replace('-', '_')}") and f.endswith(f"{timeframe}.csv")]

    raw_data = None
    if cache_files:
        # Sort by file modification time (newest first)
        cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
        latest_cache = cache_files[0]
        
        logging.info(f"Loading data from cache: {latest_cache}")
        raw_data = file_manager.load_raw_data(latest_cache, exchange='cache')
        
        # Convert DataFrame to dictionary format to match network-fetched data format
        if isinstance(raw_data, pd.DataFrame):
            # Split data based on platform column
            platforms_in_data = raw_data['platform'].unique()
            raw_data_dict = {}
            for platform in platforms_in_data:
                platform_df = raw_data[raw_data['platform'] == platform].copy()
                platform_df = platform_df.drop(columns=['platform'])  # Remove platform column
                raw_data_dict[platform] = platform_df
            raw_data = raw_data_dict
    
    # If no cache available or cache is empty, fetch from network
    if not raw_data:
        logging.info(f"Fetching data from network: {symbol}, timeframe: {timeframe}")
        data_fetcher = DataFetcher(
            platforms=platforms,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            proxy=CONFIG['data']['proxy']
        )
        raw_data = data_fetcher.fetch_data()
        
        # Save to cache
        if raw_data:
            # Generate cache filename
            cache_filename = f"{symbol.replace('-', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{timeframe}.csv"
            merged_data = data_fetcher.merge_platform_data(raw_data)
            file_manager.save_raw_data(merged_data, cache_filename, exchange='cache')
            logging.info(f"Data saved to cache: {os.path.join(cache_dir, cache_filename)}")
            
    # Basic data processing
    data_processor = DataProcessor()
    processed_df = data_processor.process_data(raw_data)
    
    # Model data processing
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
    
    # Get recent data as prediction input
    _, recent_data, _ = model_data_processor.split_data()

    # ==================== 3. Predict Future Prices ====================
    # Calculate number of time steps to predict
    timeframe_steps = {
        '1m': 60,  # 60 steps per hour
        '3m': 20,  # 20 steps per hour
        '5m': 12,  # 12 steps per hour
        '10m': 6,  # 6 steps per hour
        '15m': 4,  # 4 steps per hour
        '1h': 1,   # 1 step per hour
        '4h': 0.25,  # 0.25 steps per hour
        '1d': 1/24  # 1/24 steps per hour
    }
    
    steps_per_hour = timeframe_steps.get(timeframe, 1)
    if timeframe not in timeframe_steps:
        logging.warning(f"Unknown timeframe: {timeframe}, defaulting to 1 step per hour")
    
    # Calculate prediction steps based on time unit
    if unit == 'minutes':
        prediction_steps = int((value / 60) * steps_per_hour)
    elif unit == 'hours':
        prediction_steps = int(value * steps_per_hour)
    elif unit == 'days':
        prediction_steps = int(value * 24 * steps_per_hour)
    else:
        raise ValueError(f"Unsupported time unit: {unit}, supported units: minutes, hours, days")
    
    logging.info(f"Will predict {prediction_steps} time steps into the future")
    
    # Initialize prediction results list
    predictions = []
    current_input = recent_data['X'][-1:]  # Get most recent input sample
    
    # Predict future prices step by step
    for i in range(prediction_steps):
        # Predict next time step
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0])
        
        # Update input data (remove earliest time step, add newly predicted time step)
        if model_type == 'rnn':
            # RNN model input update
            new_input = np.copy(current_input)
            new_input[0, :-1, :] = new_input[0, 1:, :]
            
            # Create reasonable predicted values for all features
            last_values = new_input[0, -2, :]
            random_changes = np.random.normal(0, 0.001, size=last_values.shape)
            new_values = last_values + random_changes
            new_values[-1] = next_pred[0]  # Last one is predicted closing price
            
            new_input[0, -1, :] = new_values
            current_input = new_input
        else:
            # Other models' input update
            new_input = np.copy(current_input)
            feature_size = new_input.shape[1] // model_data_processor.time_steps
            new_input[0, :-feature_size] = new_input[0, feature_size:]
            new_input[0, -1] = next_pred[0]  # Update last close price
            current_input = new_input
    
    # ==================== 4. Result Processing and Visualization ====================
    # Convert prediction results back to original scale
    predictions = np.array(predictions)
    predictions_original = model_data_processor.inverse_transform_y(predictions)
    
    # Create prediction results DataFrame
    last_date = processed_df.index[-1]
    
    # Filter historical data, only show the most recent 60 minutes instead of a full day
    sixty_mins_ago = processed_df.index[-1] - pd.Timedelta(minutes=60)
    historical_data_filtered = processed_df[processed_df.index >= sixty_mins_ago]

    # timeframe to pandas frequency conversion
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
        logging.warning(f"Unknown timeframe: {timeframe}, defaulting to 1 day frequency")

    future_dates = pd.date_range(start=last_date, periods=prediction_steps+1, freq=freq)[1:]

    # Create prediction results DataFrame
    prediction_df = pd.DataFrame({
        'predicted_close': predictions_original
    }, index=future_dates)
    
    # Visualize prediction results
    visualizer = ModelVisualizer()
    visualizer.plot_future_prediction(
        historical_data=historical_data_filtered,  # Only use past day of data
        prediction_df=prediction_df, 
        model_name=model_type, 
        save=True, 
        show=False
    )
    
    logging.info(f"Prediction complete, predicted {len(prediction_df)} time points")
    return prediction_df


def predict_multiple_timeframes(model_path, time_period=None, symbol='ETH-USDT', timeframes=None):
    """
    Use a saved model to predict future price trends across multiple timeframes
    
    Parameters:
        model_path (str): Model file path
        time_period (dict): Prediction time period, format {'unit': unit, 'value': value}
                           Units can be 'minutes', 'hours', 'days'
                           Example: {'unit': 'minutes', 'value': 10} means predict 10 minutes in the future
                           If None, default predicts 1 day
        symbol (str): Trading pair symbol
        timeframes (list): List of timeframes to predict, default ['1m', '3m', '5m', '10m', '15m', '1h']
        
    Returns:
        dict: Dictionary containing prediction results for each timeframe
    """
    if timeframes is None:
        timeframes = ['5m']
    
    # If time_period is None, default predict 1 day
    if time_period is None:
        time_period = {'unit': 'days', 'value': 1}
    
    logging.info(f"Starting multi-timeframe prediction: {timeframes}, prediction time: {time_period['value']} {time_period['unit']}")
    
    results = {}
    for tf in timeframes:
        logging.info(f"Predicting for timeframe: {tf}")
        prediction_df = predict_future(model_path, time_period, symbol, tf)
        results[tf] = prediction_df
        
        # If predicting 10 minutes into the future, write result to file
        # if time_period['unit'] == 'minutes' and time_period['value'] == 10:
        #     # Get prediction for 10 minutes ahead
        #     ten_min_prediction = prediction_df.iloc[0]['predicted_close']
        #     # Write to file
        #     with open('predict-yuyao.txt', 'w') as f:
        #         f.write(f"{ten_min_prediction}")
        #     logging.info(f"10-minute prediction value {ten_min_prediction} written to predict-yuyao.txt file")
        
    return results


def predict_main(prediction_minutes=60):
    """
    预测ETH未来价格的主函数，支持自定义预测时长
    
    参数:
        prediction_minutes (int): 预测未来多少分钟的价格，默认60分钟
    
    返回:
        dict: 包含每个时间框架预测结果的字典
    """
    model_path = r'./eth_predict/models/random_forest_model_20250517_065557.pkl'
    
    # 创建自定义的时间周期字典
    time_period = {'unit': 'minutes', 'value': prediction_minutes}
    
    # 调用predict_multiple_timeframes函数进行预测
    results = predict_multiple_timeframes(
        model_path, 
        time_period=time_period,
        symbol=CONFIG['data']['symbol'], 
        timeframes=CONFIG['prediction']['timeframes']
    )
    
    logging.info(f"完成对未来{prediction_minutes}分钟的ETH价格预测")
    return results

if __name__ == "__main__":
    # 直接调用predict_main函数，可以指定预测时间
    predict_main(60)  # 预测未来60分钟
