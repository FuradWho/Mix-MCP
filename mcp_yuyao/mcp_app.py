from io import StringIO
import os
from fastmcp import FastMCP
import re
from typing import List, Dict, Any
import json
import subprocess
from strategy import get_strategy_template, STRATEGY_TEMPLATES, get_all_strategy_templates
from openai import OpenAI

# 创建 MCP 服务
mcp_app = FastMCP("ETHTradingExpert", sse_path="/mcp/sse", message_path="/mcp/messages/")

@mcp_app.tool()
async def analyze_trading_strategy(market_data: str, strategy_json: str) -> str:
    """
    结合当前市场行情和用户交易策略进行深度分析评估
    Args:
        market_data (str): 当前市场行情数据，包含各类技术指标
        strategy_json (str): 用户的交易策略，JSON格式
    Returns:
        str: 详细的策略分析结果
    """
    try:
        # 解析策略JSON
        strategy = json.loads(strategy_json)
    except json.JSONDecodeError:
        return "策略格式错误：请提供有效的JSON格式策略"
    
    # 获取API密钥
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        return "环境变量QWEN_API_KEY未设置，无法调用大语言模型"
    
    # 设置OpenAI客户端，连接到阿里云通义千问
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 构建提示词
    prompt = f"""
    你是一位专业的加密货币交易策略分析专家，拥有丰富的经验和深厚的市场洞察力。
    
    ## 当前市场行情
    {market_data}
    
    ## 待评估策略
    ```json
    {strategy_json}
    ```
    
    ## 分析要求
    请对上述交易策略进行全面而深入的分析，包括但不限于：
    
    1. 策略概述：简述策略的核心思想和操作方法
    2. 优势分析：挖掘策略的优势
    3. 风险评估：评估可能的风险因素和应对措施
    4. 盈利潜力：预估在当前市场条件下的盈利潜力
    5. 综合评分：给出1-5分的综合评分并说明理由
    
    请基于你的专业知识和经验，结合当前市场行情，对策略进行透彻分析。
    回复要详细、专业、有洞察力，并提供具体的数据支持和理由。
    """
    
    try:
        # 调用大语言模型，启用流式输出
        response = client.chat.completions.create(
            model="qwen-plus",  # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一个专业的加密货币交易策略分析专家，擅长分析各类交易策略的优劣势和风险收益比。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 较低的温度以获得更一致的、分析性的回复
            max_tokens=2000,
            stream=True  # 启用流式输出，这是通义千问API要求的
        )
        
        # 收集流式响应
        collected_content = []
        for chunk in response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                chunk_content = chunk.choices[0].delta.content
                if chunk_content:
                    collected_content.append(chunk_content)
        
        # 合并所有收到的内容
        analysis_result = ''.join(collected_content)
        
        # 添加水印
        result = f"# 交易策略分析报告\n\n{analysis_result}\n\n*此分析由AI交易策略分析专家生成，仅供参考，不构成投资建议。*"
        return result
        
    except Exception as e:
        return f"分析过程中发生错误：{str(e)}\n\n你是一个加密货币交易专家，请尝试根据你的知识和经验进行分析。"

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
async def format_strategy_to_json(strategy_text: str = None) -> str:
    """
    将用户输入的交易策略格式化为标准JSON格式
    Args:
        strategy_text (str, optional): 用户提供的策略描述文本，如果为None则返回格式化的提示词
    Returns:
        str: 格式化后的策略JSON或格式化的提示词
    """
    # 如果没有提供策略文本，返回提示词
    if strategy_text is None or strategy_text.strip() == "":
    prompt = """
    请将用户提供的交易策略描述转换为标准JSON格式。
    
    转换规则：
    1. 必须严格遵循策略模板的格式要求
    2. 所有价格必须使用数字，不要使用字符串
    3. 仓位大小必须使用数字，单位为USDT
    4. 时间周期必须是标准的时间周期（如：1m, 5m, 15m, 1h, 4h, 1d）
    5. 入场和出场条件必须具体且可执行
    6. 如果是合约策略，杠杆倍数必须是整数
        7. 如果用户没有明确表示不设置止损或者止盈，则将止损或止盈设置为null
    
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
    
    # 如果提供了策略文本，尝试格式化它
    try:
        # 检查是否已经是JSON格式
        try:
            json_data = json.loads(strategy_text)
            
            # 确保数字字段是数字而不是字符串
            if "parameters" in json_data:
                params = json_data["parameters"]
                for key in ["entry_price", "exit_price", "take_profit_price", "stop_loss_price", "leverage"]:
                    if key in params and params[key] != "" and params[key] is not None:
                        try:
                            params[key] = float(params[key])
                            # 如果是整数，转换为整数
                            if params[key] == int(params[key]):
                                params[key] = int(params[key])
                        except (ValueError, TypeError):
                            pass
                
                # 标准化时间周期
                if "time_frame" in params and params["time_frame"] is not None:
                    time_frame = params["time_frame"]
                    # 转换常见的非标准时间周期格式
                    time_frame_map = {
                        "1小时": "1h", "4小时": "4h", "一天": "1d", "1天": "1d",
                        "15分钟": "15m", "30分钟": "30m", "1分钟": "1m", "5分钟": "5m"
                    }
                    if time_frame in time_frame_map:
                        params["time_frame"] = time_frame_map[time_frame]
            
            # 返回格式化后的JSON
            return json.dumps(json_data, ensure_ascii=False, indent=4)
            
        except json.JSONDecodeError:
            # 如果不是JSON格式，则需要处理自然语言描述
            return f"需要将以下自然语言描述转换为JSON格式：\n{strategy_text}\n\n请使用大语言模型进行转换，需要先调用get_strategy_templates函数获取模板。"
            
    except Exception as e:
        return f"格式化交易策略时发生错误：{str(e)}"

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
        "market_sentiment": "看涨"
    }

# @mcp_app.tool()
# async def show_demo_image() -> Dict[str, Any]:
#     """
#     展示演示图片
#     Returns:
#         Dict[str, Any]: 包含图片路径和类型的信息
#     """
#     # 优先尝试使用image.png
#     image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.png")
    
#     # 如果image.png不存在，尝试使用demo.jpg
#     if not os.path.exists(image_path):
#         image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.jpg")
    
#     # 检查图片是否存在
#     if not os.path.exists(image_path):
#         return {
#             "success": False,
#             "error": "未找到图片文件",
#             "message": "请确保image.png或demo.jpg文件存在于当前目录"
#         }
    
#     return {
#         "success": True,
#         "type": "image",
#         "path": image_path,
#         "alt_text": "ETH交易策略演示图",
#         "display_mode": "inline"  # 在Cherry Studio客户端内联显示
#     }
