import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from models import BaseModel, MLModel, DeepLearningModel
from matplotlib import font_manager
import matplotlib as mpl
from datetime import datetime

# Set default fonts to ensure compatibility across platforms
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

class ModelVisualizer:
    """
    Model visualizer, for visualizing model training process and prediction results
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize model visualizer
        
        Parameters:
            save_dir (str): Directory to save visualization results
        """
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ModelVisualizer')
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(style="darkgrid")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                              model_name: str, save: bool = True, show: bool = True) -> None:
        """
        Plot model training history
        
        Parameters:
            history (Dict[str, List[float]]): Training history dictionary
            model_name (str): Model name
            save (bool): Whether to save the image
            show (bool): Whether to show the image
        """
        if not history.get("train_loss"):
            self.logger.warning(f"Model {model_name} has no training history")
            return
        
        self.logger.info(f"Plotting training history for model {model_name}")
        
        plt.figure(figsize=(20, 10))
        plt.plot(history["train_loss"], label="Training Loss", color="blue")
        
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss", color="red")
        
        plt.title(f"{model_name} Training History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        if save:
            filename = f"{self.save_dir}/{model_name}_training_history.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Training history plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/{model_name}_training_history.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Training history plot PDF version saved to {pdf_filename}") 
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_predictions(self, dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save: bool = True, show: bool = True) -> None:
        """
        Plot prediction results compared with actual values
        
        Parameters:
            dates (np.ndarray): Date array
            y_true (np.ndarray): Actual values
            y_pred (np.ndarray): Predicted values
            model_name (str): Model name
            save (bool): Whether to save the image
            show (bool): Whether to show the image
        """
        self.logger.info(f"Plotting prediction results for model {model_name}")
        
        plt.figure(figsize=(20, 10))
        plt.plot(dates, y_true, label="Actual Values", color="blue", linewidth=2)
        plt.plot(dates, y_pred, label="Predicted Values", color="red", linewidth=2, linestyle="--")
        
        plt.title(f"{model_name} Prediction Results")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        # Set x-axis date format
        plt.gcf().autofmt_xdate()
        
        if save:
            filename = f"{self.save_dir}/{model_name}_predictions.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Prediction results plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/{model_name}_predictions.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Prediction results plot PDF version saved to {pdf_filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, dates: np.ndarray, y_true: np.ndarray, 
                              predictions: Dict[str, np.ndarray], metrics: Dict[str, Dict[str, float]], 
                              save: bool = True, show: bool = True) -> None:
        """
        Plot comparison of multiple model prediction results
        
        Parameters:
            dates (np.ndarray): Date array
            y_true (np.ndarray): Actual values
            predictions (Dict[str, np.ndarray]): Dictionary of predicted values for each model
            metrics (Dict[str, Dict[str, float]]): Dictionary of evaluation metrics for each model
            save (bool): Whether to save the image
            show (bool): Whether to show the image
        """
        self.logger.info("Plotting comparison of multiple model prediction results")
        
        plt.figure(figsize=(20, 10))
        
        # Plot actual values
        plt.plot(dates, y_true, label="Actual Values", color="black", linewidth=3)
        
        # Plot predicted values for each model
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        linestyles = ['--', '-.', ':', '-']
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color_idx = i % len(colors)
            style_idx = i % len(linestyles)
            
            # Get RMSE metric for this model
            rmse = metrics[model_name].get('rmse', 0)
            r2 = metrics[model_name].get('r2', 0)
            
            plt.plot(dates, y_pred, 
                     label=f"{model_name} (RMSE={rmse:.4f}, R²={r2:.4f})", 
                     color=colors[color_idx], 
                     linestyle=linestyles[style_idx],
                     linewidth=2)
        
        plt.title("Multi-Model Prediction Comparison")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc='best')
        plt.grid(True)
        
        # Set x-axis date format
        plt.gcf().autofmt_xdate()
        
        # Add annotation for best model
        best_model = min(metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
        best_rmse = metrics[best_model].get('rmse', 0)
        
        plt.annotate(f"Best Model: {best_model} (RMSE={best_rmse:.4f})", 
                     xy=(0.02, 0.02), 
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        
        if save:
            filename = f"{self.save_dir}/model_comparison.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Model comparison plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/model_comparison.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Model comparison plot PDF version saved to {pdf_filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]], 
                               save: bool = True, show: bool = True) -> None:
        """
        Plot comparison of multiple model evaluation metrics
        
        Parameters:
            metrics (Dict[str, Dict[str, float]]): Dictionary of evaluation metrics for each model
            save (bool): Whether to save the image
            show (bool): Whether to show the image
        """
        self.logger.info("Plotting comparison of multiple model evaluation metrics")
        
        # Extract all models and metrics
        models = list(metrics.keys())
        all_metrics = set()
        for model_metrics in metrics.values():
            all_metrics.update(model_metrics.keys())
        
        all_metrics = sorted(list(all_metrics))
        
        # Create chart
        fig, axes = plt.subplots(len(all_metrics), 1, figsize=(12, 4*len(all_metrics)))
        if len(all_metrics) == 1:
            axes = [axes]
        
        # Create a subplot for each metric
        for i, metric_name in enumerate(all_metrics):
            metric_values = []
            for model in models:
                metric_values.append(metrics[model].get(metric_name, 0))
            
            # Plot bar chart
            bars = axes[i].bar(models, metric_values, color='skyblue')
            axes[i].set_title(f"{metric_name.upper()} Comparison")
            axes[i].set_xlabel("Model")
            axes[i].set_ylabel(metric_name.upper())
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/metrics_comparison.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Metrics comparison plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/metrics_comparison.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Metrics comparison plot PDF version saved to {pdf_filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def compare_models(self, results: Dict[str, Dict], dates = None,save: bool = True, show: bool = True) -> None:
        """
        Compare multiple model prediction results
        
        Parameters:
            results (Dict[str, Dict]): Dictionary of evaluation results for each model, including predictions and actual values
            save (bool): Whether to save the image
            show (bool): Whether to show the image
        """
        self.logger.info("Comparing multiple model prediction results")
        
        # Extract all model names
        model_names = list(results.keys())
        
        if not model_names:
            self.logger.warning("No model results to compare")
            return
        
        # Get actual values from the first model as reference
        first_model = model_names[0]
        if 'actual' not in results[first_model]:
            self.logger.warning("Missing actual values in model results")
            return
            
        y_true = results[first_model]['actual']
        
        # Create dictionaries for predictions and metrics
        predictions = {}
        metrics = {}
        
        for model_name in model_names:
            if 'predictions' in results[model_name]:
                predictions[model_name] = results[model_name]['predictions']
                
                # Extract evaluation metrics
                model_metrics = {k: v for k, v in results[model_name].items() 
                                if k not in ['predictions', 'actual'] and isinstance(v, (int, float))}
                metrics[model_name] = model_metrics
        
        # Use actual dates
        # dates = results[first_model].get('dates', np.arange(len(y_true)))
        
        # Plot model comparison
        plt.figure(figsize=(30, 9))
        
        # Plot actual values
        plt.plot(dates, y_true, label="Actual Values", color="black", linewidth=2.5)
        
        # Plot predicted values for each model
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan']
        linestyles = ['--', '-.', ':', '-']
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color_idx = i % len(colors)
            style_idx = i % len(linestyles)
            
            # Get RMSE and R2 metrics (if available)
            rmse = metrics[model_name].get('rmse', None)
            r2 = metrics[model_name].get('r2', None)
            
            label = model_name
            if rmse is not None and r2 is not None:
                label = f"{model_name} (RMSE={rmse:.4f}, R²={r2:.4f})"
            
            plt.plot(dates, y_pred, 
                     label=label, 
                     color=colors[color_idx], 
                     linestyle=linestyles[style_idx],
                     linewidth=1.5)
        
        plt.title("Multi-Model Prediction Comparison", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Predicted Value", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis date format
        plt.gcf().autofmt_xdate()
        
        # If evaluation metrics are available, find the best model
        if metrics and all('rmse' in m for m in metrics.values()):
            best_model = min(metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
            best_rmse = metrics[best_model].get('rmse', 0)
            
            plt.annotate(f"Best Model: {best_model} (RMSE={best_rmse:.4f})", 
                         xy=(0.02, 0.02), 
                         xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/models_comparison.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Models comparison plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/models_comparison.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Models comparison plot PDF version saved to {pdf_filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # Plot evaluation metrics comparison
        if metrics:
            self.plot_metrics_comparison(metrics, save=save, show=show)
            
    def plot_future_prediction(self, historical_data, prediction_df, model_name=None, save=True, show=True):
        """
        Plot historical data and future prediction data
        
        Parameters:
            historical_data (pd.DataFrame): DataFrame containing historical data with a close column
            prediction_df (pd.DataFrame): DataFrame containing prediction data with a predicted_close column
            model_name (str): Model name
            save (bool): Whether to save the chart
            show (bool): Whether to show the chart
        """
        self.logger.info("Plotting historical data and future prediction chart")
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data['close'], 
                 label='Historical Data', color='blue', linewidth=1.5)
        
        # Get last point of historical data
        last_historical_date = historical_data.index[-1]
        last_historical_value = historical_data['close'].iloc[-1]
        
        # Create a connection point to ensure smooth transition
        connection_df = pd.DataFrame({'predicted_close': [last_historical_value]}, 
                                     index=[last_historical_date])
        
        # Merge connection point with prediction data
        extended_prediction_df = pd.concat([connection_df, prediction_df])
        extended_prediction_df = extended_prediction_df.sort_index()
        
        # Plot prediction data (including connection point)
        plt.plot(extended_prediction_df.index, extended_prediction_df['predicted_close'], 
                 label='Predicted Data', color='red', linestyle='-', linewidth=1.5)
        
        # Add separation line
        plt.axvline(x=last_historical_date, color='green', linestyle='-', linewidth=1)
        
        # Add title and labels
        title = "ETH Price Prediction"
        if model_name:
            title += f" (Model: {model_name})"
        
        plt.title(title, fontsize=16)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Price (USDT)", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis date format
        plt.gcf().autofmt_xdate()
        
        # Add annotations
        plt.annotate('Historical Data', 
                     xy=(historical_data.index[len(historical_data)//2], historical_data['close'].mean()), 
                     xytext=(0, 30), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.annotate('Predicted Data', 
                     xy=(prediction_df.index[len(prediction_df)//2], prediction_df['predicted_close'].mean()), 
                     xytext=(0, -30), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/future_prediction.png"
            plt.savefig(filename, dpi=1000, bbox_inches="tight")
            self.logger.info(f"Future prediction plot saved to {filename}")
            
            # Save PDF version
            pdf_filename = f"{self.save_dir}/future_prediction.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches="tight")
            self.logger.info(f"Future prediction plot PDF version saved to {pdf_filename}")
        
        if show:
            plt.show()
        else:
            plt.close()