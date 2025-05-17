import os
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

class FileManager:
    """
    文件管理模块，用于保存和加载模型、数据和结果
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        初始化文件管理器
        
        参数:
            base_dir (str): 基础目录路径
        """
        self.base_dir = base_dir
        
        # 创建必要的目录结构
        self.raw_data_dir = os.path.join(base_dir, "raw_data")
        self.processed_data_dir = os.path.join(base_dir, "processed_data")
        self.models_dir = os.path.join(base_dir, "models")
        self.results_dir = os.path.join(base_dir, "results")
        self.logs_dir = os.path.join(base_dir, "logs")
        self.figures_dir = os.path.join(base_dir, "figures")
        
        # 确保目录存在
        for directory in [self.base_dir, self.raw_data_dir, self.processed_data_dir, 
                         self.models_dir, self.results_dir, self.logs_dir, self.figures_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('FileManager')
        
        # 设置文件处理器
        file_handler = logging.FileHandler(os.path.join(self.logs_dir, f'file_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.logger.info("文件管理器初始化完成")
    
    def save_raw_data(self, data: pd.DataFrame, filename: str, exchange: str = "generic") -> str:
        """
        保存原始数据
        
        参数:
            data (pd.DataFrame): 要保存的数据
            filename (str): 文件名
            exchange (str): 交易所名称
            
        返回:
            str: 保存的文件路径
        """
        # 创建交易所特定目录
        exchange_dir = os.path.join(self.raw_data_dir, exchange)
        os.makedirs(exchange_dir, exist_ok=True)
        
        # 构建完整文件路径
        filepath = os.path.join(exchange_dir, filename)
        
        try:
            # 确保文件扩展名为.csv
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            
            # 保存数据
            data.to_csv(filepath)
            self.logger.info(f"原始数据已保存至 {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存原始数据失败: {str(e)}")
            return ""
    
    def load_raw_data(self, filename: str, exchange: str = "generic") -> Optional[pd.DataFrame]:
        """
        加载原始数据
        
        参数:
            filename (str): 文件名
            exchange (str): 交易所名称
            
        返回:
            Optional[pd.DataFrame]: 加载的数据，如果加载失败则返回None
        """
        # 构建完整文件路径
        exchange_dir = os.path.join(self.raw_data_dir, exchange)
        filepath = os.path.join(exchange_dir, filename)
        
        # 确保文件扩展名为.csv
        if not filepath.endswith('.csv'):
            filepath += '.csv'
        
        try:
            # 加载数据
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"已从 {filepath} 加载原始数据")
            return data
        except Exception as e:
            self.logger.error(f"加载原始数据失败: {str(e)}")
            return None
    
    def save_processed_data(self, data: pd.DataFrame, filename: str, data_type: str = "generic") -> str:
        """
        保存处理后的数据
        
        参数:
            data (pd.DataFrame): 要保存的数据
            filename (str): 文件名
            data_type (str): 数据类型
            
        返回:
            str: 保存的文件路径
        """
        # 创建数据类型特定目录
        data_type_dir = os.path.join(self.processed_data_dir, data_type)
        os.makedirs(data_type_dir, exist_ok=True)
        
        # 构建完整文件路径
        filepath = os.path.join(data_type_dir, filename)
        
        try:
            # 确保文件扩展名为.csv
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            
            # 保存数据
            data.to_csv(filepath)
            self.logger.info(f"处理后的数据已保存至 {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存处理后的数据失败: {str(e)}")
            return ""
    
    def load_processed_data(self, filename: str, data_type: str = "generic") -> Optional[pd.DataFrame]:
        """
        加载处理后的数据
        
        参数:
            filename (str): 文件名
            data_type (str): 数据类型
            
        返回:
            Optional[pd.DataFrame]: 加载的数据，如果加载失败则返回None
        """
        # 构建完整文件路径
        data_type_dir = os.path.join(self.processed_data_dir, data_type)
        filepath = os.path.join(data_type_dir, filename)
        
        # 确保文件扩展名为.csv
        if not filepath.endswith('.csv'):
            filepath += '.csv'
        
        try:
            # 加载数据
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"已从 {filepath} 加载处理后的数据")
            return data
        except Exception as e:
            self.logger.error(f"加载处理后的数据失败: {str(e)}")
            return None
    
    def save_model(self, model: Any, model_name: str, model_type: str = "generic") -> str:
        """
        保存模型
        
        参数:
            model (Any): 要保存的模型
            model_name (str): 模型名称
            model_type (str): 模型类型
            
        返回:
            str: 保存的文件路径
        """
        # 创建模型类型特定目录
        model_type_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(model_type_dir, exist_ok=True)
        
        # 构建完整文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(model_type_dir, f"{model_name}_{timestamp}.pkl")
        
        try:
            # 保存模型
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            self.logger.info(f"模型已保存至 {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return ""
    
    def load_model(self, filepath: str) -> Optional[Any]:
        """
        加载模型
        
        参数:
            filepath (str): 模型文件路径
            
        返回:
            Optional[Any]: 加载的模型，如果加载失败则返回None
        """
        try:
            # 加载模型
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"已从 {filepath} 加载模型")
            return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return None
    
    def save_results(self, results: Dict, filename: str, result_type: str = "evaluation") -> str:
        """
        保存结果
        
        参数:
            results (Dict): 要保存的结果
            filename (str): 文件名
            result_type (str): 结果类型
            
        返回:
            str: 保存的文件路径
        """
        # 创建结果类型特定目录
        result_type_dir = os.path.join(self.results_dir, result_type)
        os.makedirs(result_type_dir, exist_ok=True)
        
        # 构建完整文件路径
        filepath = os.path.join(result_type_dir, filename)
        
        try:
            # 确保文件扩展名为.pkl
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
            
            # 保存结果
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"结果已保存至 {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            return ""
    
    def load_results(self, filename: str, result_type: str = "evaluation") -> Optional[Dict]:
        """
        加载结果
        
        参数:
            filename (str): 文件名
            result_type (str): 结果类型
            
        返回:
            Optional[Dict]: 加载的结果，如果加载失败则返回None
        """
        # 构建完整文件路径
        result_type_dir = os.path.join(self.results_dir, result_type)
        filepath = os.path.join(result_type_dir, filename)
        
        # 确保文件扩展名为.pkl
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        
        try:
            # 加载结果
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            self.logger.info(f"已从 {filepath} 加载结果")
            return results
        except Exception as e:
            self.logger.error(f"加载结果失败: {str(e)}")
            return None
    
    def save_figure(self, figure: plt.Figure, filename: str, figure_type: str = "generic") -> str:
        """
        保存图表
        
        参数:
            figure (plt.Figure): 要保存的图表
            filename (str): 文件名
            figure_type (str): 图表类型
            
        返回:
            str: 保存的文件路径
        """
        # 创建图表类型特定目录
        figure_type_dir = os.path.join(self.figures_dir, figure_type)
        os.makedirs(figure_type_dir, exist_ok=True)
        
        # 构建完整文件路径
        filepath = os.path.join(figure_type_dir, filename)
        
        try:
            # 确保文件扩展名为.png或.pdf
            if not (filepath.endswith('.png') or filepath.endswith('.pdf')):
                filepath += '.png'
            
            # 保存图表
            figure.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"图表已保存至 {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"保存图表失败: {str(e)}")
            return ""
    
    def list_files(self, directory: str, file_extension: str = None) -> List[str]:
        """
        列出指定目录中的文件
        
        参数:
            directory (str): 目录路径
            file_extension (str): 文件扩展名过滤器
            
        返回:
            List[str]: 文件路径列表
        """
        try:
            # 获取目录中的所有文件
            files = []
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    # 如果指定了文件扩展名，则只返回匹配的文件
                    if file_extension is None or filename.endswith(file_extension):
                        files.append(os.path.join(root, filename))
            
            return files
        except Exception as e:
            self.logger.error(f"列出文件失败: {str(e)}")
            return []
    
    def get_latest_model(self, model_type: str = "generic") -> Optional[str]:
        """
        获取最新的模型文件路径
        
        参数:
            model_type (str): 模型类型
            
        返回:
            Optional[str]: 最新模型的文件路径，如果没有找到则返回None
        """
        model_type_dir = os.path.join(self.models_dir, model_type)
        
        try:
            # 确保目录存在
            if not os.path.exists(model_type_dir):
                self.logger.warning(f"模型目录 {model_type_dir} 不存在")
                return None
            
            # 获取所有模型文件
            model_files = self.list_files(model_type_dir, '.pkl')
            
            if not model_files:
                self.logger.warning(f"在 {model_type_dir} 中没有找到模型文件")
                return None
            
            # 按文件修改时间排序
            latest_model = max(model_files, key=os.path.getmtime)
            self.logger.info(f"找到最新的模型文件: {latest_model}")
            
            return latest_model
        except Exception as e:
            self.logger.error(f"获取最新模型失败: {str(e)}")
            return None
    
    def clean_old_files(self, directory: str, max_files: int = 10, file_extension: str = None) -> None:
        """
        清理旧文件，保留最新的文件
        
        参数:
            directory (str): 目录路径
            max_files (int): 要保留的最大文件数
            file_extension (str): 文件扩展名过滤器
        """
        try:
            # 获取目录中的所有文件
            files = self.list_files(directory, file_extension)
            
            if len(files) <= max_files:
                self.logger.info(f"目录 {directory} 中的文件数量未超过限制，无需清理")
                return
            
            # 按文件修改时间排序
            files.sort(key=os.path.getmtime, reverse=True)
            
            # 删除旧文件
            for file_to_delete in files[max_files:]:
                try:
                    os.remove(file_to_delete)
                    self.logger.info(f"已删除旧文件: {file_to_delete}")
                except Exception as e:
                    self.logger.error(f"删除文件 {file_to_delete} 失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"清理旧文件失败: {str(e)}")
