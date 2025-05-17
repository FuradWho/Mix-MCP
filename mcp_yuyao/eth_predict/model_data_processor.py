import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class ModelDataProcessor:
    """
    模型数据处理模块，用于将处理后的数据转换为适合机器学习和深度学习模型使用的格式
    """
    
    def __init__(self, data: pd.DataFrame, train_ratio: float = 0.7, test_ratio: float = 0.2, 
                 validation_ratio: float = 0.1, features: List[str] = None, target: str = 'close', 
                 time_steps: int = 30, smooth: bool = False, smooth_window: int = 5):
        """
        初始化模型数据处理器
        
        参数:
            data (pd.DataFrame): 输入数据
            train_ratio (float): 训练集比例
            test_ratio (float): 测试集比例
            validation_ratio (float): 验证集比例
            features (List[str]): 特征列表
            target (str): 目标变量
            time_steps (int): 时间步长（用于序列模型）
            smooth (bool): 是否对数据进行平滑处理
            smooth_window (int): 平滑窗口大小
        """
        self.data = data.copy()
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        
        # 确保比例之和为1
        total_ratio = train_ratio + test_ratio + validation_ratio
        print(total_ratio)
        if abs(total_ratio - 1.0) > 1e-10:
            self.train_ratio = train_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
            self.validation_ratio = validation_ratio / total_ratio
        
        # 如果未指定特征，则使用所有数值列作为特征
        if features is None:
            self.features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # 排除目标变量
            if target in self.features:
                self.features.remove(target)
        else:
            self.features = features
        
        self.target = target
        self.time_steps = time_steps
        self.smooth = smooth
        self.smooth_window = smooth_window
        
        # 数据缩放器
        self.scalers = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ModelDataProcessor')
        
        # 初始化处理
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        预处理数据
        """
        self.logger.info("开始预处理数据...")
        
        # 确保索引是datetime类型
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.logger.warning("索引不是datetime类型，尝试转换...")
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                self.logger.error(f"转换索引到datetime类型失败: {str(e)}")
        
        # 检查特征是否存在于数据中
        for feature in self.features:
            if feature not in self.data.columns:
                self.logger.warning(f"特征 {feature} 不在数据列中，将被忽略")
                self.features.remove(feature)
        
        # 检查目标变量是否存在
        if self.target not in self.data.columns:
            self.logger.error(f"目标变量 {self.target} 不在数据列中")
            raise ValueError(f"目标变量 {self.target} 不在数据列中")
        
        # 如果需要平滑数据
        if self.smooth:
            self.logger.info(f"使用窗口大小 {self.smooth_window} 平滑数据...")
            for col in self.features + [self.target]:
                self.data[col] = self.data[col].rolling(window=self.smooth_window, min_periods=1).mean()
        
        self.logger.info("数据预处理完成")
    
    def split_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        将数据分割为训练集、测试集和验证集
        
        返回:
            Tuple[Dict, Dict, Dict]: 训练集、测试集和验证集的字典，每个字典包含 'X', 'y' 和 'dates' 键
        """
        self.logger.info("开始分割数据...")
        
        # 按时间排序
        data_sorted = self.data.sort_index()
        
        # 计算分割点
        n = len(data_sorted)
        train_end = int(n * self.train_ratio)
        validation_start = train_end
        validation_end = validation_start + int(n * self.validation_ratio)
        
        # 分割数据：训练集在前，验证集在中间，测试集在后
        train_data = data_sorted.iloc[:train_end]
        validation_data = data_sorted.iloc[validation_start:validation_end]
        test_data = data_sorted.iloc[validation_end:]
        
        self.logger.info(f"数据分割完成: 训练集 {len(train_data)} 行, 验证集 {len(validation_data)} 行, 测试集 {len(test_data)} 行")
        
        # 准备模型输入数据
        train_dict = None if self.train_ratio == 0.0 else self._prepare_model_data(train_data, is_train=True)
        validation_dict = None if self.validation_ratio == 0.0 else self._prepare_model_data(validation_data, is_train=False)
        test_dict = None if self.test_ratio == 0.0 else self._prepare_model_data(test_data, is_train=False)
        
        return train_dict, test_dict, validation_dict
    
    def _prepare_model_data(self, data: pd.DataFrame, is_train: bool = False) -> Dict:
        """
        准备模型输入数据
        
        参数:
            data (pd.DataFrame): 输入数据
            is_train (bool): 是否为训练数据
            
        返回:
            Dict: 包含 'X', 'y' 和 'dates' 的字典
        """
        X = data[self.features].values
        y = data[self.target].values
        dates = data.index
        
        # 缩放数据
        if is_train or 'X' not in self.scalers:  # 即使不是训练集，如果scalers不存在也创建
            # 创建新的缩放器
            self.scalers['X'] = MinMaxScaler(feature_range=(0, 1))
            self.scalers['y'] = MinMaxScaler(feature_range=(0, 1))
            
            X_scaled = self.scalers['X'].fit_transform(X)
            y_scaled = self.scalers['y'].fit_transform(y.reshape(-1, 1)).flatten()
        else:
            # 使用已有的缩放器
            X_scaled = self.scalers['X'].transform(X)
            y_scaled = self.scalers['y'].transform(y.reshape(-1, 1)).flatten()
        
        # 创建时间序列数据（如果需要）
        if self.time_steps > 0:
            X_seq, y_seq, dates_seq = self._create_sequences(X_scaled, y_scaled, dates)
            return {'X': X_seq, 'y': y_seq, 'dates': dates_seq}
        else:
            return {'X': X_scaled, 'y': y_scaled, 'dates': dates}
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        创建时间序列数据
        
        参数:
            X (np.ndarray): 特征数据
            y (np.ndarray): 目标数据
            dates (pd.DatetimeIndex): 日期索引
            
        返回:
            Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]: 序列化的特征、目标和日期
        """
        self.logger.info(f"创建时间序列数据，时间步长: {self.time_steps}")
        
        X_seq, y_seq, dates_seq = [], [], []
        
        for i in range(len(X) - self.time_steps):
            X_seq.append(X[i:i + self.time_steps])
            y_seq.append(y[i + self.time_steps])
            dates_seq.append(dates[i + self.time_steps])
        
        return np.array(X_seq), np.array(y_seq), pd.DatetimeIndex(dates_seq)
    
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        将缩放后的目标变量转换回原始尺度
        
        参数:
            y_scaled (np.ndarray): 缩放后的目标变量
            
        返回:
            np.ndarray: 原始尺度的目标变量
        """
        return self.scalers['y'].inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        返回:
            List[str]: 特征名称列表
        """
        return self.features
    
    def get_target_name(self) -> str:
        """
        获取目标变量名称
        
        返回:
            str: 目标变量名称
        """
        return self.target
    
    def create_lagged_features(self, lag_periods: List[int] = [1, 3, 5, 7]) -> None:
        """
        创建滞后特征
        
        参数:
            lag_periods (List[int]): 滞后周期列表
        """
        self.logger.info(f"创建滞后特征，周期: {lag_periods}")
        
        for col in self.features + [self.target]:
            for lag in lag_periods:
                lag_col = f"{col}_lag_{lag}"
                self.data[lag_col] = self.data[col].shift(lag)
                
                # 将新创建的滞后特征添加到特征列表中
                if lag_col not in self.features and col != self.target:
                    self.features.append(lag_col)
        
        # 删除包含NaN的行
        self.data = self.data.dropna()
        self.logger.info(f"滞后特征创建完成，数据形状: {self.data.shape}")
    
    def create_rolling_features(self, windows: List[int] = [3, 7, 14, 30], 
                               functions: Dict[str, callable] = None) -> None:
        """
        创建滚动窗口特征
        
        参数:
            windows (List[int]): 窗口大小列表
            functions (Dict[str, callable]): 函数字典，键为函数名，值为函数对象
        """
        if functions is None:
            functions = {'mean': np.mean, 'std': np.std}
            
        self.logger.info(f"创建滚动窗口特征，窗口大小: {windows}")
        
        for col in self.features + [self.target]:
            for window in windows:
                for func_name, func in functions.items():
                    roll_col = f"{col}_{func_name}_{window}"
                    self.data[roll_col] = self.data[col].rolling(window=window, min_periods=1).apply(func)
                    
                    # 将新创建的滚动特征添加到特征列表中
                    if roll_col not in self.features and col != self.target:
                        self.features.append(roll_col)
        
        self.logger.info(f"滚动窗口特征创建完成，数据形状: {self.data.shape}")
    
    def create_technical_indicators(self) -> None:
        """
        创建技术指标特征
        """
        self.logger.info("创建技术指标特征...")
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            self.logger.warning(f"缺少创建技术指标所需的列: {missing_cols}")
            return
        
        # 相对强弱指标 (RSI)
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # 移动平均线 (MA)
        for window in [5, 10, 20, 50]:
            self.data[f'MA_{window}'] = self.data['close'].rolling(window=window).mean()
        
        # 指数移动平均线 (EMA)
        for window in [5, 10, 20, 50]:
            self.data[f'EMA_{window}'] = self.data['close'].ewm(span=window, adjust=False).mean()
        
        # 布林带 (Bollinger Bands)
        self.data['BB_middle'] = self.data['close'].rolling(window=20).mean()
        self.data['BB_std'] = self.data['close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + 2 * self.data['BB_std']
        self.data['BB_lower'] = self.data['BB_middle'] - 2 * self.data['BB_std']
        
        # MACD
        exp1 = self.data['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_hist'] = self.data['MACD'] - self.data['MACD_signal']
        
        # 添加新创建的技术指标到特征列表中
        tech_indicators = ['RSI', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 
                          'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                          'BB_middle', 'BB_upper', 'BB_lower',
                          'MACD', 'MACD_signal', 'MACD_hist']
        
        for indicator in tech_indicators:
            if indicator not in self.features:
                self.features.append(indicator)
        
        # 删除包含NaN的行
        self.data = self.data.dropna()
        self.logger.info(f"技术指标特征创建完成，数据形状: {self.data.shape}")
    
    def filter_features(self, correlation_threshold: float = 0.1) -> None:
        """
        根据与目标变量的相关性过滤特征
        
        参数:
            correlation_threshold (float): 相关性阈值
        """
        self.logger.info(f"根据与目标变量的相关性过滤特征，阈值: {correlation_threshold}")
        
        # 计算特征与目标变量的相关性
        correlations = self.data[self.features + [self.target]].corr()[self.target]
        
        # 过滤掉相关性低于阈值的特征
        low_corr_features = correlations[abs(correlations) < correlation_threshold].index.tolist()
        
        # 从特征列表中移除低相关性特征
        for feature in low_corr_features:
            if feature in self.features:
                self.features.remove(feature)
        
        self.logger.info(f"特征过滤完成，保留 {len(self.features)} 个特征")
    
    def save_processed_data(self, filename: str) -> None:
        """
        保存处理后的数据
        
        参数:
            filename (str): 文件名
        """
        try:
            self.data.to_csv(filename)
            self.logger.info(f"处理后的数据已保存至 {filename}")
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
    
    def load_processed_data(self, filename: str) -> None:
        """
        加载处理后的数据
        
        参数:
            filename (str): 文件名
        """
        try:
            self.data = pd.read_csv(filename, index_col=0, parse_dates=True)
            self.logger.info(f"已从 {filename} 加载数据，数据形状: {self.data.shape}")
            
            # 重新初始化处理
            self._preprocess_data()
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
