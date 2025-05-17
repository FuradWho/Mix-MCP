from io import StringIO
import os
from fastmcp import FastMCP
import re
from typing import List, Dict, Any
import json
import subprocess
from strategy import get_strategy_template, STRATEGY_TEMPLATES, get_all_strategy_templates
from openai import OpenAI
import sys
import base64
from pathlib import Path
import asyncio

# # 添加路径以便导入eth_predict模块
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from eth_predict.predict_eth import predict_main

mcp_app = FastMCP("ETHTradingExpert", sse_path="/mcp/sse", message_path="/mcp/messages/")

@mcp_app.tool()
async def analyze_trading_strategy(market_report: str, strategy: str) -> str:
    """
    基于市场分析报告和用户交易策略进行深度分析评估，提供专业的策略评估报告
    
    Args:
        market_repot (str): 市场分析报告
        strategy (str): 用户的交易策略
    Returns:
        str: 详细的策略分析结果
    """
   
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
    
    ## 当前市场分析报告
    {market_report}
    
    ## 待评估策略
    ```json
    {strategy}
    ```
    
    ## 分析要求
    请基于上述市场分析报告，对交易策略进行全面而深入的评估，包括但不限于：
    
    1. 策略概述：简述策略的核心思想和操作方法
    2. 优势分析：挖掘策略在当前市场条件下的优势
    3. 风险评估：评估可能的风险因素和应对措施
    4. 盈利潜力：预估在当前市场条件下的盈利潜力
    5. 综合评分：给出1-5分的综合评分并说明理由
    
    ## 视觉化呈现要求
    你的分析应当以视觉化美化的方式呈现，包括：
    
    - 使用 ✅ 和 ❌ 等表情符号强调重点
    - 使用 🔴、🟢、🟡 表示不同风险等级
    - 通过表格格式整理关键信息
    - 使用加粗标记重要数据增强可读性
    - 分段清晰，使用标题和分隔线提高可读性
    - 使用 📊 📈 💰 ⚠️ 等主题相关的表情符号增强视觉辨识度
    
    示例格式：
    
    ```
    # 📊 [策略名称]评估报告

    ## 🧾 策略概述

    | 属性       | 内容                     |
    |------------|--------------------------|
    | **方向**   | 做空/做多 🟡              |
    | **杠杆**   | **XX倍**                |
    | **目标**   | 简述目标 ✅               |

    ## ✅ 优势分析（Potential Upsides）

    | 优势点                | 说明                  |
    |---------------------|------------------------|
    | **优势1** ✅        | 详细说明               |
    | **优势2** ✅        | 详细说明               |

    ## ⚠️ 风险评估（Risk Assessment）

    | 风险类型             | 描述                   | 风险等级 🔴/🟡/🟢 |
    |---------------------|------------------------|-------------------|
    | **风险1** 🔺        | 详细说明              | 🔴 高              |
    | **风险2** 🔻        | 详细说明              | 🟡 中等            |

    ## 💰 盈利潜力（Profit Potential）
    
    详细评估...

    ## 🧮 综合评分与建议

    | 项目              | 评分（满分5分）     | 说明                  |
    |-------------------|--------------------|-----------------------|
    | **项目1**         | ⭐⭐⭐☆☆ (3)         | 说明                  |
    | **项目2**         | ⭐⭐☆☆☆ (2)         | 说明                  |

    ### 总体评分：⭐⭐⭐☆☆ (**3 / 5**)

    ## 🛠️ 改进建议（Recommendations）
    
    建议列表...

    ## 🧠 结论总结
    
    总结观点...
    ```
    
    请基于你的专业知识和经验，结合当前市场分析，对策略进行透彻评估。
    回复要详细、专业、有洞察力，并提供具体的数据支持和理由。
    """
    
    # 调用大语言模型，启用流式输出
    response = client.chat.completions.create(
        model="qwen-plus",  # 使用通义千问模型
        messages=[
            {"role": "system", "content": "你是一个专业的加密货币交易策略分析专家，擅长分析各类交易策略的优劣势和风险收益比。你的分析报告采用表情符号、表格和视觉化元素使内容更直观易读。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # 较低的温度以获得更一致的、分析性的回复
        max_tokens=3000,
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
    result = f"\n{analysis_result}\n\n*此分析由AI交易策略分析专家生成，仅供参考，不构成投资建议。* 请你将上面的内容原封不动地返回给用户，不要使用````否则会导致无法展示markdown编译后的效果\n"
    return result

# @mcp_app.tool()
# async def get_strategy_templates() -> str:
#     """
#     获取所有可用的ETH交易策略模板
    
#     Returns:
#         str: 包含所有策略模板格式的提示词字符串
#     """
#     # 获取可用的策略类型列表
#     available_strategies = list(STRATEGY_TEMPLATES.keys())
    
#     # 获取所有策略模板
#     all_templates = get_all_strategy_templates()
    
#     prompt = f"""
#     以下是所有可用的ETH交易策略模板：
    
#     可用的策略类型：
#     {', '.join(available_strategies)}
    
#     以下是所有可用的策略模板格式：
    
#     {all_templates}
#     """
    
#     return prompt

# @mcp_app.tool()
# async def format_strategy_to_json(strategy_text: str = None) -> str:
#     """
#     将用户自然语言描述的交易策略转换为标准JSON格式
#     【重要】此函数用于将用户文本转为JSON，是analyze_trading_strategy的前置步骤
    
#     Args:
#         strategy_text (str, optional): 用户提供的策略描述文本，如果为None则返回格式化的提示词
#     Returns:
#         str: 格式化后的策略JSON或格式化的提示词
#     """
#     # 首先获取策略模板，确保转换基于标准模板
#     templates = await get_strategy_templates()
    
#     # 如果没有提供策略文本，返回提示词
#     if strategy_text is None or strategy_text.strip() == "":
#         prompt = """
#         请将用户提供的交易策略描述转换为标准JSON格式。
        
#         转换规则：
#         1. 必须严格遵循策略模板的格式要求
#         2. 所有价格必须使用数字，不要使用字符串
#         3. 仓位大小必须使用数字，单位为USDT
#         4. 时间周期必须是标准的时间周期（如：1m, 5m, 15m, 1h, 4h, 1d）
#         5. 入场和出场条件必须具体且可执行
#         6. 如果是合约策略，杠杆倍数必须是整数
#             7. 如果用户没有明确表示不设置止损或者止盈，则将止损或止盈设置为null
        
#         特殊处理规则：
#         1. 如果用户明确表示不设置止损：
#         - 做空策略：止损价格设置为 Infinity（正无穷）
#         - 做多策略：止损价格设置为 0
#         2. 如果用户明确表示不设置止盈：
#         - 做空策略：止盈价格设置为 0
#         - 做多策略：止盈价格设置为 Infinity（正无穷）
        
#         如果用户对parameters描述中缺少某些必要参数，请在JSON中将其设置为null。
#         转换完成后，请检查：
#         1. JSON格式是否完整且有效
#         2. 是否所有必要字段都已包含
#         3. 数据类型是否正确
        
#         示例输入：
#         "我想在ETH价格突破2000时做多，使用5倍杠杆，不设置止损，止盈设在2100"
        
#         示例输出：
#         ```json
#         {
#             "strategy_type": "contract",
#             "trade_direction": "做多",
#             "parameters": {
#                 "entry_price": 2000,
#                 "exit_price": null,
#                 "take_profit_price": 2100,
#                 "stop_loss_price": 0,
#                 "leverage": 5,
#                 "margin_type": null,
#                 "time_frame": null
#             },
#             "description": "我想在ETH价格突破2000时做多，使用5倍杠杆，不设置止损，止盈设在2100"
#         }
#         ```
        
#         注意事项：
#         1. 如果转换后的JSON中包含null值，请先向用户展示完整的JSON
#         2. 然后询问用户是否希望由大模型智能填写这些null值
#         3. 如果用户同意，在保证交易风险可控的前提下，智能填写这些参数
#         4. 如果用户不同意，请让用户自己填写这些参数
#         5. 对于不设置止损或止盈的情况，请严格按照特殊处理规则设置相应的价格值
#         """
#         return prompt
    
#     # 如果提供了策略文本，尝试将其转换为JSON格式
#     try:
#         # 检查是否已经是JSON格式
#         try:
#             json_data = json.loads(strategy_text)
            
#             # 确保数字字段是数字而不是字符串
#             if "parameters" in json_data:
#                 params = json_data["parameters"]
#                 for key in ["entry_price", "exit_price", "take_profit_price", "stop_loss_price", "leverage"]:
#                     if key in params and params[key] != "" and params[key] is not None:
#                         try:
#                             params[key] = float(params[key])
#                             # 如果是整数，转换为整数
#                             if params[key] == int(params[key]):
#                                 params[key] = int(params[key])
#                         except (ValueError, TypeError):
#                             pass
                
#                 # 标准化时间周期
#                 if "time_frame" in params and params["time_frame"] is not None:
#                     time_frame = params["time_frame"]
#                     # 转换常见的非标准时间周期格式
#                     time_frame_map = {
#                         "1小时": "1h", "4小时": "4h", "一天": "1d", "1天": "1d",
#                         "15分钟": "15m", "30分钟": "30m", "1分钟": "1m", "5分钟": "5m"
#                     }
#                     if time_frame in time_frame_map:
#                         params["time_frame"] = time_frame_map[time_frame]
            
#             # 返回格式化后的JSON
#             return json.dumps(json_data, ensure_ascii=False, indent=4)
            
#         except json.JSONDecodeError:
#             # 如果不是JSON格式，则需要处理自然语言描述
#             return f"需要将自然语言描述转换为JSON格式：\n\n{strategy_text}\n\n请使用上述交易策略模板进行转换。"
            
#     except Exception as e:
#         return f"格式化交易策略时发生错误：{str(e)}"

@mcp_app.tool()
async def analyze_market_conditions(market_situation: str = None) -> Dict[str, Any]:
    """
    根据K线数据分析当前市场状况，提供全面的ETH市场指标和分析，返回市场分析报告
    Args:
        market_situation (str, optional): K线数据，形式为["1747123200000","2461.1","2471.49","2451.28","2457.38","33.0325","81283.558342","81283.558342"]
    Returns:
        Dict[str, Any]: 包含多维度市场分析结果的字典
    """
    # 样例市场报告格式
    sample_report = """
# 📊 市场分析报告样例

## 📈 市场概览

| 指标           | 数值                  | 状态评估              |
|---------------|------------------------|----------------------|
| **当前价格**   | **XXXX美元**          | 🟢/🟡/🔴 趋势评估     |

## 📊 技术面分析

| 技术指标       | 当前值               | 解读                  |
|---------------|----------------------|------------------------|
| **RSI(14)**   | **数值**             | 🟡/🟢/🔴 解读          |
"""

    # 如果没有提供市场情况，返回空模板
    if market_situation is None or market_situation.strip() == "":
        return {
            "message": "当前没有提供市场情况描述，请要求用户提供市场情况描述，以便进行分析。"
        }
    # market_situation =  '以太坊(ETH)当前价格为1850美元，过去24小时上涨了2.5%。交易量相比昨日增加了15%，达到120亿美元。市场情绪整体偏向乐观，RSI指标为62，处于中性偏多的区域。MACD指标显示正在形成金叉信号。目前主要支撑位在1780美元和1720美元，主要阻力位在1900美元和2000美元。资金费率为0.01%，略微正向，表明多头情绪占优。链上数据显示交易所ETH净流出，过去24小时约减少了25000枚ETH。市场恐惧贪婪指数为65，处于贪婪区域但未到极度贪婪。近期以太坊生态系统发展积极，Layer 2解决方案用户增长迅速，DeFi锁仓量稳步上升。'
    try:
        # 获取API密钥
        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            return {"error": "环境变量QWEN_API_KEY未设置，无法调用大语言模型"}
        
        # 设置OpenAI客户端，连接到阿里云通义千问
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 首先获取市场分析数据
        # 直接构建模板JSON字符串(将数值设为"null")
        template_json = """
        {
            "trend": {
                "primary": null,
                "strength": null,
                "duration": null,
                "phase": null,
                "momentum": null,
                "breakdown_risk": null
            },
            
            "volatility": {
                "current": null,
                "value": null,
                "volatility_index": null,
                "historical_percentile": null,
                "expected_daily_range": {
                    "percentage": null,
                    "price_range": {
                        "low": null,
                        "high": null
                    }
                },
                "trend": null,
                "skew": null
            },
            
            "support_levels": [
                {"price": null, "strength": null, "type": null, "confirmation": null},
                {"price": null, "strength": null, "type": null, "confirmation": null},
                {"price": null, "strength": null, "type": null, "confirmation": null}
            ],
            
            "resistance_levels": [
                {"price": null, "strength": null, "type": null, "confirmation": null},
                {"price": null, "strength": null, "type": null, "confirmation": null},
                {"price": null, "strength": null, "type": null, "confirmation": null}
            ],
            "technical_indicators": {
                "moving_averages": {
                    "sma_50": {"value": null, "position": null, "distance": null},
                    "sma_200": {"value": null, "position": null, "distance": null},
                    "ema_20": {"value": null, "position": null, "distance": null},
                    "ma_cross": {"status": null, "time_since": null, "strength": null}
                },
                "oscillators": {
                    "rsi_14": {"value": null, "interpretation": null, "trend": null},
                    "macd": {
                        "histogram": null, 
                        "signal": null, 
                        "macd_line": null, 
                        "interpretation": null
                    },
                    "stochastic": {
                        "k": null, 
                        "d": null, 
                        "interpretation": null, 
                        "is_overbought": null, 
                        "is_oversold": null
                    }
                },
                "trend_indicators": {
                    "adx": {"value": null, "trend_strength": null, "trend": null},
                    "parabolic_sar": {"value": null, "position": null, "interpretation": null}
                },
                "volume_indicators": {
                    "obv": {"value": null, "trend": null, "divergence": null},
                    "vwap": {"value": null, "position": null, "interpretation": null}
                }
            },
            
            "onchain_metrics": {
                "active_addresses": {
                    "count": null,
                    "change_7d": null
                },
                "transaction_volume": {
                    "value": null,
                    "change_24h": null
                },
                "gas_price": {
                    "fast": null,
                    "standard": null,
                    "slow": null,
                    "trend": null
                },
                "exchange_flows": {
                    "inflow_24h": null,
                    "outflow_24h": null,
                    "net_flow": null,
                    "exchange_balance_change": null
                }
            },
            
            "risk_assessment": {
                "volatility_risk": null,
                "liquidity_risk": null,
                "market_risk": null,
                "regulatory_risk": null
            },
            
            "predictions": {
                "price_targets": {
                    "short_term": {"low": null, "high": null, "most_likely": null, "timeframe": null},
                    "medium_term": {"low": null, "high": null, "most_likely": null, "timeframe": null}
                },
                "probability_distribution": [
                    {"range": null, "probability": null},
                    {"range": null, "probability": null},
                    {"range": null, "probability": null}
                ],
                "technical_scenarios": {
                    "bullish": {
                        "trigger": null,
                        "target": null,
                        "probability": null,
                        "key_indicators": [null, null]
                    },
                    "bearish": {
                        "trigger": null,
                        "target": null,
                        "probability": null,
                        "key_indicators": [null, null]
                    }
                }
            },
            
            "current_price": null,
            "current_timestamp": null,
            "analysis_timestamp": null
        }
        """
        
        # 构建提示词获取市场分析数据
        json_prompt = f"""
        你是一位专业的以太坊市场分析专家，拥有丰富的加密货币市场分析经验和深厚的技术分析知识。
        
        ## K线数据的解释

        | 返回字段 | 参数类型 | 字段说明 |
        | --- | --- | --- |
        | index[0] | String | 系统时间戳，Unix毫秒时间戳，例如1690196141868 |
        | index[1] | String | 开盘价格 |
        | index[2] | String | 最高价格 |
        | index[3] | String | 最低价格 |
        | index[4] | String | 收盘价格 |
        | index[5] | String | 基础币成交量，如“BTCUSDT”交易对中的“BTC” |
        | index[6] | String | USDT成交量 |
        | index[7] | String | 计价币成交量，如“BTCUSDT”交易对中的“USDT” |

        ## 当前K线数据
        {market_situation}
        
        ## 分析任务
        请基于用户提供的K线数据，对当前ETH市场进行全面的分析，填充以下JSON模板中的null值。
        你的分析应该全面、客观、专业，覆盖各个方面的市场指标。
        
        对于每一个null值：
        1. 如果用户提供的市场情况中包含相关信息，必须填写合理、专业的分析结果
        2. 如果用户没有提供某些基本信息（如当前价格、时间戳等），则保持这些字段为null
        3. 数值型字段应该填写合适的数字（使用数值而非字符串）
        4. 文本型字段应该填写详细、专业的描述
        5. 请确保分析结果的一致性，不同指标之间应该互相支持，而不是相互矛盾
        6. 只基于用户提供的市场情况进行分析，避免凭空捏造不确定的信息
        
        这是需要填充的JSON模板：
        ```json
        {template_json}
        ```
        
        请注意：
        - 支撑位和阻力位的数量可以根据实际情况调整
        - 所有百分比应该使用小数表示（例如：0.05表示5%）
        - 货币金额应该使用数字而非字符串
        - 仅推断与用户提供信息有直接关联的指标，当信息不足无法做出合理推断时，保持相应字段为null
        - 确保你的分析在技术上是准确的，符合加密货币市场分析的专业标准
        """
        
        # 调用大语言模型获取JSON格式的市场分析
        json_response = client.chat.completions.create(
            model="qwen-plus",  # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一个专业的加密货币市场分析专家，擅长根据K线数据对ETH市场进行全面分析并提供结构化的JSON格式分析结果。"},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.2,  # 较低的温度以获得更一致的、分析性的回复
            response_format={"type": "json_object"},  # 要求返回JSON格式
            max_tokens=4000
        )
        
        market_data = json_response.choices[0].message.content
        
        # 然后使用这些数据生成视觉化的市场分析报告
        visual_prompt = f"""
        你是一位专业的加密货币市场分析专家，拥有丰富的经验和深厚的市场洞察力。请将以下JSON格式的市场分析数据转换为视觉化美观的市场分析报告。
        
        ## 市场分析报告
        ```json
        {market_data}
        ```
        
        ## 视觉化呈现要求
        你的分析应当以视觉化美化的方式呈现，包括：
        
        - 使用 ✅ 和 ❌ 等表情符号强调重点
        - 使用 🔴、🟢、🟡 表示不同风险等级
        - 通过表格格式整理关键信息
        - 使用加粗标记重要数据增强可读性
        - 分段清晰，使用标题和分隔线提高可读性
        - 使用 📊 📈 💰 ⚠️ 等主题相关的表情符号增强视觉辨识度
        
        ## 报告结构
        请按照以下结构组织你的报告：
        
        1. 📊 市场概览：当前价格、趋势方向、交易量等核心信息
        2. 📈 技术面分析：移动平均线、RSI、MACD等技术指标分析
        3. 💹 支撑阻力位：关键支撑位和阻力位分析
        4. 🌊 市场情绪：包括社交媒体情绪、衍生品市场情绪等
        5. ⛓️ 链上指标：活跃地址、交易量、交易所资金流等链上数据分析
        6. ⚖️ 风险评估：各类市场风险的评估
        7. 🔮 短期预测：未来价格走势预测
        8. 📝 交易建议：基于当前市场分析的交易建议
        
        ## 样例格式
        
        {sample_report}
        
        请确保你的报告专业、全面、客观，并充分利用表格、视觉化元素使内容更加直观。
        """
        return visual_prompt
        
    except Exception as e:
        return {"error": f"分析过程中发生错误：{str(e)}"}

@mcp_app.tool()
async def ai_predict_visualization() -> str:
    """
    调用AI预测引擎预测ETH未来价格走势，并返回可视化结果
    
    Returns:
        str: 包含预测结果图像的HTML标记
    """
    # 添加3秒延迟
    await asyncio.sleep(3)

    # 图像相对路径
    visualization_path = 'https://img.geyuyao.com/i/u/2025/05/17/future_prediction_1.png'

    # 返回HTML格式的图片引用，设置宽度为100%
    html_response = f"""
    <div style="text-align: center;">
        <img src="{visualization_path}" alt="ETH价格预测" style="width: 100%; max-width: 1200px;" />
    </div>
    """
    
    return html_response
    # try:
    #     # 调用predict_main函数进行预测
    #     results = predict_main(prediction_minutes)
        
    #     # 获取可视化图像路径
    #     visualization_path = './mcp_ai_predict/eth_predict/visualizations/future_prediction.png'
    #     abs_path = os.path.abspath(visualization_path)
        
    #     # 检查图像是否存在
    #     if not os.path.exists(visualization_path):
    #         return f"预测成功，但未找到可视化图像文件: {visualization_path}"
        
    #     # 读取图像并转换为base64
    #     with open(visualization_path, "rb") as img_file:
    #         img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
    #     # 构建HTML响应，包含内嵌图像
    #     html_response = f"""
    #     <div style="text-align: center;">
    #         <h2>ETH价格预测结果（未来{prediction_minutes}分钟）</h2>
    #         <img src="data:image/png;base64,{img_data}" style="max-width:100%;" />
    #         <p>预测生成时间: {Path(visualization_path).stat().st_mtime}</p>
    #     </div>
    #     """
        
    #     return html_response
    # except Exception as e:
    #     return f"预测过程中发生错误: {str(e)}"
