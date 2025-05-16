from typing import Dict, Any

# 现货交易策略模板
SPOT_STRATEGY_TEMPLATE = """
{
    "strategy_type": "spot",
    "trade_direction": "做多/做空",
    "parameters": {
        "entry_price": "入场价格",
        "exit_price": "出场价格",
        "take_profit_price": "止盈价格",
        "stop_loss_price": "止损价格",
        "time_frame": "交易时间周期"
    },
    "description": "策略描述"
}
"""

# 合约交易策略模板
CONTRACT_STRATEGY_TEMPLATE = """
{
    "strategy_type": "contract",
    "trade_direction": "做多/做空",
    "parameters": {
        "entry_price": "入场价格",
        "exit_price": "出场价格",
        "take_profit_price": "止盈价格",
        "stop_loss_price": "止损价格",
        "leverage": "杠杆倍数",
        "margin_type": "保证金类型(全仓/逐仓)",
        "time_frame": "交易时间周期"
    },
    "description": "策略描述"
}
"""

# 策略类型映射
STRATEGY_TEMPLATES: Dict[str, str] = {
    "spot": SPOT_STRATEGY_TEMPLATE,
    "contract": CONTRACT_STRATEGY_TEMPLATE
}

def get_strategy_template(strategy_type: str) -> str:
    """
    获取指定类型的策略模板
    Args:
        strategy_type (str): 策略类型 ("spot" 或 "contract")
    Returns:
        str: 策略模板
    """
    return STRATEGY_TEMPLATES.get(strategy_type, SPOT_STRATEGY_TEMPLATE)

def get_all_strategy_templates() -> str:
    """
    获取所有策略模板的组合
    Returns:
        str: 所有策略模板的组合字符串
    """
    templates = []
    for strategy_type, template in STRATEGY_TEMPLATES.items():
        templates.append(f"=== {strategy_type.upper()} 策略模板 ===\n{template}\n")
    
    return "\n".join(templates)
