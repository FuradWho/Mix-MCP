import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import logging

class DataFetcher:
    """
    数据获取模块，用于从不同交易平台获取加密货币的历史数据
    """
    
    def __init__(self, platforms, symbol, start_date, end_date, timeframe='1d', proxy=None):
        """
        初始化数据获取器
        
        参数:
            platforms (list): 交易平台列表，如 ['okx', 'bitget', 'binance']
            symbol (str): 交易对，如 'ETH-USDT'
            start_date (datetime 或 int): 开始日期，可以是datetime对象或年份
            end_date (datetime 或 int): 结束日期，可以是datetime对象或年份
            timeframe (str): 时间周期，如 '1d', '1h', '15m' 等
            proxy (dict): 代理设置，如 {'http': 'http://localhost:7897', 'https': 'http://localhost:7897'}
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('DataFetcher')
        
        self.platforms = platforms
        self.symbol = symbol
        self.proxy = proxy
        
        # 处理开始日期和结束日期
        if isinstance(start_date, int):
            self.start_date = datetime(start_date, 1, 1)
        else:
            self.start_date = start_date
            
        if isinstance(end_date, int):
            self.end_date = datetime(end_date, 12, 31)
        else:
            self.end_date = end_date
            
        self.timeframe = timeframe
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """
        初始化交易所API连接
        """
        for platform in self.platforms:
            try:
                exchange_class = getattr(ccxt, platform.lower())
                exchange_options = {}
                
                # 只有当proxy不为None时才设置代理
                if self.proxy is not None:
                    exchange_options['proxies'] = self.proxy
                
                exchange_instance = exchange_class(exchange_options)
                self.exchanges[platform] = exchange_instance
            except AttributeError:
                self.logger.warning(f"不支持的交易平台: {platform}")
            except Exception as e:
                self.logger.error(f"初始化交易平台 {platform} 失败: {str(e)}")
    
    def _format_symbol(self, platform, symbol):
        """
        根据不同平台格式化交易对
        
        参数:
            platform (str): 交易平台名称
            symbol (str): 原始交易对
            
        返回:
            str: 格式化后的交易对
        """
        # 不同平台可能有不同的交易对格式
        if '-' in symbol:
            base, quote = symbol.split('-')
        else:
            base, quote = symbol.split('/')
            
        if platform.lower() == 'okx':
            return f"{base}/{quote}"  # 修正okx平台的交易对格式为 BTC/USDT
        elif platform.lower() == 'binance':
            return f"{base}/{quote}"
        elif platform.lower() == 'bitget':
            return f"{base}/{quote}"
        else:
            return symbol
    
    def _fetch_ohlcv_data(self, exchange, symbol, timeframe, since, limit=1000):
        """
        获取OHLCV数据
        
        参数:
            exchange (ccxt.Exchange): 交易所对象
            symbol (str): 交易对
            timeframe (str): 时间周期
            since (int): 开始时间戳（毫秒）
            limit (int): 单次获取的最大数据量
            
        返回:
            list: OHLCV数据列表
        """
        try:
            # 确保交易所支持获取OHLCV数据
            if not exchange.has['fetchOHLCV']:
                self.logger.error(f"{exchange.id} 不支持获取OHLCV数据")
                return []
                
            # 获取市场数据
            markets = exchange.load_markets()
            if symbol not in markets:
                self.logger.warning(f"{symbol} 在 {exchange.id} 上不可用")
                return []
                
            # 获取数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            return ohlcv
        except ccxt.NetworkError as e:
            self.logger.error(f"网络错误 ({exchange.id}): {e}")
            return []
        except ccxt.ExchangeError as e:
            self.logger.error(f"交易所错误 ({exchange.id}): {e}")
            return []
        except Exception as e:
            self.logger.error(f"从 {exchange.id} 获取数据失败: {str(e)}")
            return []
    
    def fetch_data(self):
        """
        从所有配置的交易平台获取历史数据
        
        返回:
            dict: 包含各平台数据的字典，格式为 {platform: dataframe}
        """
        all_data = {}
        
        for platform, exchange in self.exchanges.items():
            self.logger.info(f"正在从 {platform} 获取 {self.symbol} 的历史数据...")
            
            try:
                formatted_symbol = self._format_symbol(platform, self.symbol)
                since = int(self.start_date.timestamp() * 1000)
                end_timestamp = int(self.end_date.timestamp() * 1000)
                
                all_ohlcv = []
                max_retries = 3
                
                while since < end_timestamp:
                    success = False
                    retry_count = 0
                    
                    while not success and retry_count < max_retries:
                        try:
                            # 确保交易所已初始化
                            if not exchange.has['fetchOHLCV']:
                                raise ValueError(f"{platform} 不支持获取OHLCV数据")
                            
                            # 加载市场数据
                            if not hasattr(exchange, 'markets') or not exchange.markets:
                                exchange.load_markets()
                            
                            # 获取数据
                            ohlcv = exchange.fetch_ohlcv(
                                formatted_symbol,
                                self.timeframe,
                                since,
                                limit=1000  # 明确指定limit
                            )
                            
                            if ohlcv and len(ohlcv) > 0:
                                all_ohlcv.extend(ohlcv)
                                since = ohlcv[-1][0] + 1
                                success = True
                            else:
                                self.logger.warning(f"获取到空数据，时间戳: {since}")
                                retry_count += 1
                                time.sleep(2)
                        except Exception as e:
                            self.logger.error(f"获取数据失败 (重试 {retry_count + 1}/{max_retries}): {str(e)}")
                            retry_count += 1
                            time.sleep(2)
                    
                    if not success:
                        # 如果重试失败，记录错误并继续
                        self.logger.error(f"在时间戳 {since} 处获取数据失败")
                        since += self._get_timeframe_milliseconds(self.timeframe)
                
                # 检查是否获取到数据
                if not all_ohlcv:
                    self.logger.error(f"从 {platform} 未获取到任何数据")
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # 确保所有必要的列都存在
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in df.columns:
                        self.logger.error(f"缺少必要的列: {col}")
                        return {}
                
                # 数据清理和验证
                df = df.sort_index()
                df = df.loc[self.start_date:self.end_date]
                
                # 检查数据是否为空
                if df.empty:
                    self.logger.error(f"处理后的数据为空")
                    continue
                
                all_data[platform] = df
                self.logger.info(f"从 {platform} 成功获取了 {len(df)} 条数据")
                
            except Exception as e:
                self.logger.error(f"处理 {platform} 数据时出错: {str(e)}")
                continue
        
        # 最终检查
        if not all_data:
            self.logger.error("未能从任何平台获取数据")
            return {}
        
        return all_data
    
    def _get_timeframe_milliseconds(self, timeframe):
        """计算时间间隔的毫秒数"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60 * 1000
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60 * 60 * 1000
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60 * 60 * 1000
        return 60 * 1000  # 默认1分钟
    
    def _fill_data_gaps(self, df):
        """填补数据间隙"""
        if len(df) == 0:
            return df
        
        # 创建完整的时间索引
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self.timeframe.replace('m', 'min').replace('h', 'H').replace('d', 'D')
        )
        
        # 重新索引并使用插值填充
        df = df.reindex(full_index)
        
        # 对于较大的间隙（比如超过5个时间单位），使用None标记
        gap_threshold = 5
        for col in df.columns:
            mask = df[col].isna()
            if mask.any():
                gap_sizes = mask.astype(int).groupby(mask.ne(mask.shift()).cumsum()).sum()
                large_gaps = gap_sizes[gap_sizes > gap_threshold].index
                for gap_id in large_gaps:
                    gap_mask = mask.groupby(mask.ne(mask.shift()).cumsum()) == gap_id
                    df.loc[gap_mask, col] = None
        
        return df
    
    def fetch_latest_data(self, days=7):
        """
        获取最近几天的数据
        
        参数:
            days (int): 获取最近几天的数据
            
        返回:
            dict: 包含各平台最新数据的字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        temp_start_date = self.start_date
        temp_end_date = self.end_date
        
        # 临时修改日期范围
        self.start_date = start_date
        self.end_date = end_date
        
        # 获取数据
        latest_data = self.fetch_data()
        
        # 恢复原始日期范围
        self.start_date = temp_start_date
        self.end_date = temp_end_date
        
        return latest_data
    
    def merge_platform_data(self, data_dict):
        """
        合并不同平台的数据
        
        参数:
            data_dict (dict): 包含各平台数据的字典
            
        返回:
            DataFrame: 合并后的数据
        """
        if not data_dict:
            self.logger.warning("没有数据可合并")
            return pd.DataFrame()
        
        # 如果只有一个平台，直接返回该平台的数据
        if len(data_dict) == 1:
            platform, df = next(iter(data_dict.items()))
            df = df.copy()
            df['platform'] = platform
            return df
        
        # 多平台数据合并逻辑保持不变
        dfs = []
        for platform, df in data_dict.items():
            df_copy = df.copy()
            df_copy['platform'] = platform
            dfs.append(df_copy)
        
        merged_df = pd.concat(dfs)
        merged_df.sort_index(inplace=True)
        return merged_df
    
    def save_data_to_csv(self, data, filename):
        """
        将数据保存为CSV文件
        
        参数:
            data (DataFrame 或 dict): 要保存的数据
            filename (str): 文件名
        """
        try:
            if isinstance(data, dict):
                # 如果是字典，先合并数据
                merged_data = self.merge_platform_data(data)
                merged_data.to_csv(filename)
                self.logger.info(f"合并数据已保存至 {filename}")
            else:
                # 如果是DataFrame，直接保存
                data.to_csv(filename)
                self.logger.info(f"数据已保存至 {filename}")
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
    
    def _fetch_missing_data(self, exchange, symbol, start_timestamp, end_timestamp):
        """
        获取缺失的数据
        
        参数:
            exchange: 交易所对象
            symbol: 交易对符号
            start_timestamp: 开始时间戳（毫秒）
            end_timestamp: 结束时间戳（毫秒）
            
        返回:
            list: OHLCV数据列表
        """
        missing_data = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                # 获取这段时间的数据
                ohlcv = self._fetch_ohlcv_data(exchange, symbol, self.timeframe, current_timestamp)
                
                if ohlcv and len(ohlcv) > 0:
                    # 只添加在时间范围内的数据
                    valid_data = [x for x in ohlcv if start_timestamp <= x[0] < end_timestamp]
                    missing_data.extend(valid_data)
                    
                    # 更新时间戳
                    if ohlcv[-1][0] >= end_timestamp:
                        break
                    current_timestamp = ohlcv[-1][0] + 1
                else:
                    # 如果没有获取到数据，向前移动一个时间单位
                    current_timestamp += self._get_timeframe_milliseconds(self.timeframe)
                
                # 避免请求过于频繁
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                self.logger.error(f"获取缺失数据时出错: {str(e)}")
                # 发生错误时，向前移动一个时间单位
                current_timestamp += self._get_timeframe_milliseconds(self.timeframe)
        
        return missing_data
