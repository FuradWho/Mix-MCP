import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional

class DataProcessor:
    """
    数据处理模块，用于对从交易平台获取的原始数据进行基础处理
    """
    
    def __init__(self):
        """
        初始化数据处理器
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('DataProcessor')
    
    def process_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        处理从不同平台获取的原始数据
        
        参数:
            data_dict (Dict[str, pd.DataFrame]): 包含各平台数据的字典，格式为 {platform: dataframe}
            
        返回:
            pd.DataFrame: 处理后的数据
        """
        if not data_dict:
            self.logger.warning("没有数据可处理")
            return pd.DataFrame()
        
        # 合并不同平台的数据
        merged_df = self._merge_platform_data(data_dict)
        
        # 处理缺失值
        processed_df = self._handle_missing_values(merged_df)
        
        # 添加技术指标
        processed_df = self._add_technical_indicators(processed_df)
        
        # 添加时间特征
        processed_df = self._add_time_features(processed_df)
        
        # 标准化数据
        processed_df = self._normalize_data(processed_df)
        
        self.logger.info(f"数据处理完成，处理后数据形状: {processed_df.shape}")
        
        return processed_df
    
    def _merge_platform_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并不同平台的数据
        
        参数:
            data_dict (Dict[str, pd.DataFrame]): 包含各平台数据的字典
            
        返回:
            pd.DataFrame: 合并后的数据
        """
        self.logger.info("合并不同平台数据...")
        
        # 创建一个空的DataFrame列表
        dfs = []
        
        for platform, df in data_dict.items():
            # 添加平台标识列
            df_copy = df.copy()
            df_copy['platform'] = platform
            dfs.append(df_copy)
        
        # 合并所有DataFrame
        merged_df = pd.concat(dfs)
        
        # 按时间排序
        merged_df.sort_index(inplace=True)
        
        self.logger.info(f"合并完成，合并后数据形状: {merged_df.shape}")
        
        return merged_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 处理后的数据
        """
        self.logger.info("处理缺失值...")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            self.logger.info(f"发现缺失值: \n{missing_values[missing_values > 0]}")
            
            # 对于OHLCV数据，使用前向填充方法处理缺失值
            df_filled = df.fillna(method='ffill')
            
            # 如果仍有缺失值（例如序列开始处），使用后向填充
            df_filled = df_filled.fillna(method='bfill')
            
            # 检查是否还有缺失值
            remaining_missing = df_filled.isnull().sum().sum()
            if remaining_missing > 0:
                self.logger.warning(f"填充后仍有 {remaining_missing} 个缺失值，将删除这些行")
                df_filled = df_filled.dropna()
            
            self.logger.info("缺失值处理完成")
            return df_filled
        else:
            self.logger.info("数据中没有缺失值")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 添加技术指标后的数据
        """
        self.logger.info("添加技术指标...")
        result_df = df.copy()
        
        # 检查是否只有一个平台
        platforms = result_df['platform'].unique()
        if len(platforms) == 1:
            # 直接计算技术指标，不进行分组
            result_df['MA5'] = result_df['close'].rolling(window=5).mean()
            result_df['MA10'] = result_df['close'].rolling(window=10).mean()
            result_df['MA20'] = result_df['close'].rolling(window=20).mean()
            
            # RSI计算
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            result_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算布林带 (Bollinger Bands)
            result_df['BB_middle'] = result_df['close'].rolling(window=20).mean()
            result_df['BB_std'] = result_df['close'].rolling(window=20).std()
            result_df['BB_upper'] = result_df['BB_middle'] + 2 * result_df['BB_std']
            result_df['BB_lower'] = result_df['BB_middle'] - 2 * result_df['BB_std']
            
            # 计算MACD
            exp1 = result_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = result_df['close'].ewm(span=26, adjust=False).mean()
            result_df['MACD'] = exp1 - exp2
            result_df['MACD_signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()
            result_df['MACD_hist'] = result_df['MACD'] - result_df['MACD_signal']
            
            # 计算成交量变化率
            result_df['volume_change'] = result_df['volume'].pct_change()
            
            # 计算价格变化率
            result_df['price_change'] = result_df['close'].pct_change()
            
            # 计算波动率 (Volatility)
            result_df['volatility'] = result_df['close'].rolling(window=20).std() / result_df['close'].rolling(window=20).mean()
        else:
            # 原有的分组处理逻辑
            for platform, group in result_df.groupby('platform'):
                # 计算移动平均线 (MA)
                result_df.loc[group.index, 'MA5'] = group['close'].rolling(window=5).mean()
                result_df.loc[group.index, 'MA10'] = group['close'].rolling(window=10).mean()
                result_df.loc[group.index, 'MA20'] = group['close'].rolling(window=20).mean()
                
                # 计算相对强弱指标 (RSI)
                delta = group['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                result_df.loc[group.index, 'RSI'] = 100 - (100 / (1 + rs))
                
                # 计算布林带 (Bollinger Bands)
                result_df.loc[group.index, 'BB_middle'] = group['close'].rolling(window=20).mean()
                result_df.loc[group.index, 'BB_std'] = group['close'].rolling(window=20).std()
                result_df.loc[group.index, 'BB_upper'] = result_df.loc[group.index, 'BB_middle'] + 2 * result_df.loc[group.index, 'BB_std']
                result_df.loc[group.index, 'BB_lower'] = result_df.loc[group.index, 'BB_middle'] - 2 * result_df.loc[group.index, 'BB_std']
                
                # 计算MACD
                exp1 = group['close'].ewm(span=12, adjust=False).mean()
                exp2 = group['close'].ewm(span=26, adjust=False).mean()
                result_df.loc[group.index, 'MACD'] = exp1 - exp2
                result_df.loc[group.index, 'MACD_signal'] = result_df.loc[group.index, 'MACD'].ewm(span=9, adjust=False).mean()
                result_df.loc[group.index, 'MACD_hist'] = result_df.loc[group.index, 'MACD'] - result_df.loc[group.index, 'MACD_signal']
                
                # 计算成交量变化率
                result_df.loc[group.index, 'volume_change'] = group['volume'].pct_change()
                
                # 计算价格变化率
                result_df.loc[group.index, 'price_change'] = group['close'].pct_change()
                
                # 计算波动率 (Volatility)
                result_df.loc[group.index, 'volatility'] = group['close'].rolling(window=20).std() / group['close'].rolling(window=20).mean()
        
        # 删除NaN值（由于计算技术指标产生的）
        result_df = result_df.dropna()
        
        self.logger.info(f"技术指标添加完成，数据形状: {result_df.shape}")
        
        return result_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 添加时间特征后的数据
        """
        self.logger.info("添加时间特征...")
        
        # 创建一个新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 确保索引是datetime类型
        if not isinstance(result_df.index, pd.DatetimeIndex):
            self.logger.warning("索引不是datetime类型，尝试转换...")
            try:
                result_df.index = pd.to_datetime(result_df.index)
            except Exception as e:
                self.logger.error(f"转换索引到datetime类型失败: {str(e)}")
                return df
        
        # 添加日期特征
        result_df['year'] = result_df.index.year
        result_df['month'] = result_df.index.month
        result_df['day'] = result_df.index.day
        result_df['dayofweek'] = result_df.index.dayofweek
        result_df['quarter'] = result_df.index.quarter
        
        # 添加是否为周末特征
        result_df['is_weekend'] = result_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 添加月初/月末特征
        result_df['is_month_start'] = result_df.index.is_month_start.astype(int)
        result_df['is_month_end'] = result_df.index.is_month_end.astype(int)
        
        # 添加季度初/季度末特征
        result_df['is_quarter_start'] = result_df.index.is_quarter_start.astype(int)
        result_df['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
        
        self.logger.info(f"时间特征添加完成，数据形状: {result_df.shape}")
        
        return result_df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据
        
        参数:
            df (pd.DataFrame): 输入数据
            
        返回:
            pd.DataFrame: 标准化后的数据
        """
        self.logger.info("标准化数据...")
        
        # 创建一个新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 需要标准化的数值列
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 按平台分组标准化
        for platform, group in result_df.groupby('platform'):
            for col in numeric_cols:
                if col in result_df.columns:
                    # 计算z-score标准化
                    mean = group[col].mean()
                    std = group[col].std()
                    if std != 0:  # 避免除以零
                        result_df.loc[group.index, f'{col}_norm'] = (group[col] - mean) / std
                    else:
                        result_df.loc[group.index, f'{col}_norm'] = 0
                        self.logger.warning(f"列 {col} 在平台 {platform} 的标准差为零，无法进行标准化")
        
        self.logger.info(f"数据标准化完成，数据形状: {result_df.shape}")
        
        return result_df
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        重采样数据到指定频率
        
        参数:
            df (pd.DataFrame): 输入数据
            freq (str): 重采样频率，如 'D'（每日）, 'W'（每周）, 'M'（每月）
            
        返回:
            pd.DataFrame: 重采样后的数据
        """
        self.logger.info(f"将数据重采样为 {freq} 频率...")
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("索引不是datetime类型，尝试转换...")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                self.logger.error(f"转换索引到datetime类型失败: {str(e)}")
                return df
        
        # 按平台分组重采样
        resampled_dfs = []
        
        for platform, group in df.groupby('platform'):
            # 选择数值列进行重采样
            numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
            
            # 从numeric_cols中移除platform列（如果存在）
            if 'platform' in numeric_cols:
                numeric_cols.remove('platform')
            
            # 定义重采样规则
            resampled = group[numeric_cols].resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 添加平台信息
            resampled['platform'] = platform
            
            resampled_dfs.append(resampled)
        
        # 合并所有重采样后的数据
        if resampled_dfs:
            result_df = pd.concat(resampled_dfs)
            result_df.sort_index(inplace=True)
            
            self.logger.info(f"重采样完成，数据形状: {result_df.shape}")
            return result_df
        else:
            self.logger.warning("重采样失败，返回原始数据")
            return df
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date: Union[str, datetime], 
                             end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        按日期范围过滤数据
        
        参数:
            df (pd.DataFrame): 输入数据
            start_date (str 或 datetime): 开始日期
            end_date (str 或 datetime): 结束日期
            
        返回:
            pd.DataFrame: 过滤后的数据
        """
        self.logger.info(f"按日期范围过滤数据: {start_date} 至 {end_date}")
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("索引不是datetime类型，尝试转换...")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                self.logger.error(f"转换索引到datetime类型失败: {str(e)}")
                return df
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 过滤数据
        filtered_df = df.loc[start_date:end_date]
        
        self.logger.info(f"日期过滤完成，过滤后数据形状: {filtered_df.shape}")
        
        return filtered_df
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        计算不同周期的收益率
        
        参数:
            df (pd.DataFrame): 输入数据
            periods (List[int]): 收益率周期列表
            
        返回:
            pd.DataFrame: 添加收益率后的数据
        """
        self.logger.info(f"计算收益率，周期: {periods}")
        
        # 创建一个新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 按平台分组计算收益率
        for platform, group in result_df.groupby('platform'):
            for period in periods:
                result_df.loc[group.index, f'return_{period}d'] = group['close'].pct_change(periods=period)
        
        self.logger.info(f"收益率计算完成，数据形状: {result_df.shape}")
        
        return result_df
    
    def smooth_data(self, df: pd.DataFrame, columns: List[str], window: int = 5, method: str = 'rolling') -> pd.DataFrame:
        """
        平滑数据
        
        参数:
            df (pd.DataFrame): 输入数据
            columns (List[str]): 需要平滑的列
            window (int): 窗口大小
            method (str): 平滑方法，'rolling'（移动平均）或 'ewm'（指数加权移动平均）
            
        返回:
            pd.DataFrame: 平滑后的数据
        """
        self.logger.info(f"使用 {method} 方法平滑数据，窗口大小: {window}")
        
        # 创建一个新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 按平台分组平滑数据
        for platform, group in result_df.groupby('platform'):
            for col in columns:
                if col in result_df.columns:
                    if method == 'rolling':
                        result_df.loc[group.index, f'{col}_smooth'] = group[col].rolling(window=window, min_periods=1).mean()
                    elif method == 'ewm':
                        result_df.loc[group.index, f'{col}_smooth'] = group[col].ewm(span=window, min_periods=1).mean()
                    else:
                        self.logger.warning(f"不支持的平滑方法: {method}，跳过平滑")
        
        self.logger.info(f"数据平滑完成，数据形状: {result_df.shape}")
        
        return result_df

