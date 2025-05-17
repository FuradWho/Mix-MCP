from io import StringIO
import os
from fastmcp import FastMCP
import re
from typing import List, Dict, Any
import json
import subprocess
from strategy import get_strategy_template, STRATEGY_TEMPLATES, get_all_strategy_templates
from openai import OpenAI

mcp_app = FastMCP("ETHTradingExpert", sse_path="/mcp/sse", message_path="/mcp/messages/")

@mcp_app.tool()
async def analyze_trading_strategy(market_data: str, strategy_json: str) -> str:
    """
    基于市场分析结果和用户交易策略进行深度分析评估，提供专业的策略评估报告
    【重要】使用前请先调用 analyze_market_conditions 获取市场分析数据作为 market_data 参数
    
    Args:
        market_data (str): 市场分析数据，是 analyze_market_conditions 函数的返回结果
        strategy_json (str): 用户的交易策略，JSON格式，通过 format_strategy_to_json 函数生成
    Returns:
        str: 详细的策略分析结果
    """
    
    try:
        # 解析策略JSON，但不再严格校验格式
        strategy = json.loads(strategy_json)
    except:
        strategy = strategy_json
    
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
    {market_data}
    
    ## 待评估策略
    ```json
    {strategy_json}
    ```
    
    ## 分析要求
    请基于上述市场分析报告，对交易策略进行全面而深入的评估，包括但不限于：
    
    1. 策略概述：简述策略的核心思想和操作方法
    2. 优势分析：挖掘策略在当前市场条件下的优势
    3. 风险评估：评估可能的风险因素和应对措施
    4. 盈利潜力：预估在当前市场条件下的盈利潜力
    5. 综合评分：给出1-5分的综合评分并说明理由
    
    请基于你的专业知识和经验，结合当前市场分析，对策略进行透彻评估。
    回复要详细、专业、有洞察力，并提供具体的数据支持和理由。
    """
    
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
    分析当前市场状况，提供全面的ETH市场指标和分析，返回市场分析报告
    Args:
        market_situation (str, optional): 用户描述的市场情况，如果为None则返回空模板
    Returns:
        Dict[str, Any]: 包含多维度市场分析结果的字典
    """
    # 如果没有提供市场情况，返回空模板
    # if market_situation is None or market_situation.strip() == "":
    #     return {
    #         "message": "当前没有提供市场情况描述，请要求用户提供市场情况描述，以便进行分析。"
    #     }
    market_situation =  '以太坊(ETH)当前价格为1850美元，过去24小时上涨了2.5%。交易量相比昨日增加了15%，达到120亿美元。市场情绪整体偏向乐观，RSI指标为62，处于中性偏多的区域。MACD指标显示正在形成金叉信号。目前主要支撑位在1780美元和1720美元，主要阻力位在1900美元和2000美元。资金费率为0.01%，略微正向，表明多头情绪占优。链上数据显示交易所ETH净流出，过去24小时约减少了25000枚ETH。市场恐惧贪婪指数为65，处于贪婪区域但未到极度贪婪。近期以太坊生态系统发展积极，Layer 2解决方案用户增长迅速，DeFi锁仓量稳步上升。'
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
            
            "market_sentiment": {
                "overall": null,
                "strength": null,
                "fear_greed_index": null,
                "social_sentiment": {
                    "twitter": null,
                    "reddit": null,
                    "weibo": null,
                    "telegram": null
                },
                "derivatives_sentiment": {
                    "funding_rate": null,
                    "funding_rate_trend": null,
                    "long_short_ratio": null,
                    "liquidations_24h": {
                        "long": null,
                        "short": null
                    },
                    "open_interest": {
                        "value": null,
                        "change_24h": null
                    }
                }
            },
            
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
        
        # 构建提示词
        prompt = f"""
        你是一位专业的以太坊市场分析专家，拥有丰富的加密货币市场分析经验和深厚的技术分析知识。
        
        ## 当前市场情况描述
        {market_situation}
        
        ## 分析任务
        请基于用户提供的市场情况描述，对当前ETH市场进行全面的分析，填充以下JSON模板中的null值。
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
        
        # 调用大语言模型
        response = client.chat.completions.create(
            model="qwen-plus",  # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一个专业的加密货币市场分析专家，擅长对ETH市场进行全面分析并提供结构化的JSON格式分析结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 较低的温度以获得更一致的、分析性的回复
            response_format={"type": "json_object"},  # 要求返回JSON格式
            max_tokens=4000
        )
        
        # 解析返回的JSON
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            try:
                # 尝试解析JSON
                analysis_result = json.loads(content)
                return analysis_result
            except json.JSONDecodeError:
                # 如果无法解析JSON，返回原始内容
                return content
        else:
            return {"error": "大模型返回结果格式错误"}
            
    except Exception as e:
        return {"error": f"分析过程中发生错误：{str(e)}"}
