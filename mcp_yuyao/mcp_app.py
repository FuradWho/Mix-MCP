from io import StringIO
import os
from fastmcp import FastMCP
import re
from typing import List, Dict, Any
import json
import subprocess
from strategy import get_strategy_template, STRATEGY_TEMPLATES, get_all_strategy_templates

# 创建 MCP 服务
mcp_app = FastMCP("ETHTradingExpert", sse_path="/mcp/sse", message_path="/mcp/messages/")

@mcp_app.tool()
async def query_strategy_info(query_subject: str) -> str:
    """
    查询关于特定主题的信息
    Args:
        query_subject (str): 用户查询的问题主语，可以是任何加密货币相关的主题或概念
    Returns:
        str: 查询结果信息
    """
  
    # 构建查询命令，先激活conda环境，切换目录，再执行主命令
    query = f"{query_subject}是什么"
    
    # 构建完整命令：激活conda环境 -> 切换目录 -> 执行查询
    cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml && cd /Users/yuyaoge/Project/Mix-MCP/graphrag && graphrag query --root ./ragtest --method local --query \"{query}\""
    
    try:
        # 使用subprocess.run确保等待命令执行完成，使用bash确保conda命令可以执行
        process = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True, executable='/bin/bash')
        
        # 获取命令的输出
        result = process.stdout.strip()
        
        # 如果有内容返回，添加前缀
        if result:
            return f"你是一个加密货币交易专家，熟知各种交易策略。\n{result}"
        else:
            return "你是一个加密货币交易专家，熟知各种交易策略。"
            
    except subprocess.CalledProcessError as e:
        # 命令执行出错
        return "你是一个加密货币交易专家，熟知各种交易策略。"
    except Exception as e:
        # 其他异常
        return "你是一个加密货币交易专家，熟知各种交易策略。"

@mcp_app.tool()
async def evaluate_strategy(strategy: str) -> Dict[str, Any]:
    """
    **在调用此函数之前，必须先调用过get_strategy_templates函数**
    评估用户提供的交易策略
    Args:
        strategy (str): 用户提供的策略描述
    Returns:
        Dict[str, Any]: 策略评估结果
    """
    # 这里可以添加实际的策略评估逻辑
    return {
        "score": 85,
        "strengths": ["优势1", "优势2"],
        "weaknesses": ["劣势1", "劣势2"],
        "risk_assessment": "风险评估结果",
        "suggestions": ["改进建议1", "改进建议2"]
    }

@mcp_app.tool()
async def get_strategy_templates() -> str:
    """
    获取所有可用的交易策略模板
    Returns:
        str: 包含所有策略模板格式的提示词字符串
    """
    # 获取可用的策略类型列表
    available_strategies = list(STRATEGY_TEMPLATES.keys())
    
    # 获取所有策略模板
    all_templates = get_all_strategy_templates()
    
    prompt = f"""
    以下是所有可用的ETH交易策略模板：
    
    可用的策略类型：
    {', '.join(available_strategies)}
    
    以下是所有可用的策略模板格式：
    
    {all_templates}
    """
    
    return prompt

@mcp_app.tool()
async def format_strategy_to_json() -> str:
    """
    **在调用此函数之前，必须先调用过get_strategy_templates函数**
    将用户输入的交易策略格式化为标准JSON格式
    * 此函数用于将用户提供的非结构化策略描述转换为标准JSON格式
    * 转换后的JSON必须符合get_strategy_templates中定义的模板格式
    * 如果某些必要参数缺失，将返回null值
    Returns:
        str: 格式化策略的提示词
    """
    prompt = """
    请将用户提供的交易策略描述转换为标准JSON格式。
    
    转换规则：
    1. 必须严格遵循策略模板的格式要求
    2. 所有价格必须使用数字，不要使用字符串
    3. 仓位大小必须使用数字，单位为USDT
    4. 时间周期必须是标准的时间周期（如：1m, 5m, 15m, 1h, 4h, 1d）
    5. 入场和出场条件必须具体且可执行
    6. 如果是合约策略，杠杆倍数必须是整数
    
    特殊处理规则：
    1. 如果用户明确表示不设置止损：
       - 做空策略：止损价格设置为 Infinity（正无穷）
       - 做多策略：止损价格设置为 0
    2. 如果用户明确表示不设置止盈：
       - 做空策略：止盈价格设置为 0
       - 做多策略：止盈价格设置为 Infinity（正无穷）
    
    如果用户对parameters描述中缺少某些必要参数，请在JSON中将其设置为null。
    转换完成后，请检查：
    1. JSON格式是否完整且有效
    2. 是否所有必要字段都已包含
    3. 数据类型是否正确
    
    示例输入：
    "我想在ETH价格突破2000时做多，使用5倍杠杆，不设置止损，止盈设在2100"
    
    示例输出：
    ```json
    {
        "strategy_type": "contract",
        "trade_direction": "做多",
        "parameters": {
            "entry_price": 2000,
            "exit_price": null,
            "take_profit_price": 2100,
            "stop_loss_price": 0,
            "leverage": 5,
            "margin_type": null,
            "time_frame": null
        },
        "description": "我想在ETH价格突破2000时做多，使用5倍杠杆，不设置止损，止盈设在2100"
    }
    ```
    
    注意事项：
    1. 如果转换后的JSON中包含null值，请先向用户展示完整的JSON
    2. 然后询问用户是否希望由大模型智能填写这些null值
    3. 如果用户同意，在保证交易风险可控的前提下，智能填写这些参数
    4. 如果用户不同意，请让用户自己填写这些参数
    5. 对于不设置止损或止盈的情况，请严格按照特殊处理规则设置相应的价格值
    """
    
    return prompt

@mcp_app.tool()
async def analyze_market_conditions() -> Dict[str, Any]:
    """
    分析当前市场状况
    Returns:
        Dict[str, Any]: 市场分析结果
    """
    return {
        "trend": "上升趋势",
        "volatility": "中等",
        "support_levels": ["1800", "1750"],
        "resistance_levels": ["1900", "1950"],
        "market_sentiment": "看涨",
        "recommended_strategies": ["合约网格", "现货马丁格尔"]
    }

@mcp_app.tool()
async def show_demo_image() -> Dict[str, Any]:
    """
    展示演示图片
    Returns:
        Dict[str, Any]: 包含图片路径和类型的信息
    """
    # 优先尝试使用image.png
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
    
    # 如果image.png不存在，尝试使用demo.jpg
    if not os.path.exists(image_path):
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.jpg")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": "未找到图片文件",
            "message": "请确保image.png或demo.jpg文件存在于当前目录"
        }
    
    return {
        "success": True,
        "type": "image",
        "path": image_path,
        "alt_text": "ETH交易策略演示图",
        "display_mode": "inline"  # 在Cherry Studio客户端内联显示
    }
