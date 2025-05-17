# ETH交易策略AI预测系统

这是一个基于MCP（大模型通信协议）的ETH交易策略AI预测系统，可以对ETH价格进行预测并提供交易策略分析。

## 功能特点

- 使用多种机器学习模型预测ETH价格走势
- 分析用户交易策略并提供专业评估
- 生成可视化的市场分析报告
- 通过SSE（Server-Sent Events）格式实时传输数据

## 快速启动 (使用uv)

### 前置条件

确保已安装以下工具：
- Python 3.12+
- uv (Python包管理工具)

如果尚未安装uv，可以使用以下命令安装：

```bash
curl -sSf https://install.python-uv.org | python3
```

### 环境配置

1. 进入项目目录:

```bash
cd mcp_ai_predict
```

2. 使用uv创建虚拟环境并安装依赖:

```bash
uv venv
uv pip install -e .
```

或者你也可以直接从requirements.txt安装:

```bash
uv venv
uv pip install -r requirements.txt
```

3. 激活虚拟环境:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. 设置环境变量:

创建.env文件并添加必要的API密钥:

```bash
echo "QWEN_API_KEY=你的百炼API密钥" > .env
```

### 运行服务

启动服务:

```bash
python main.py --port 8001 --transport sse
```

服务将在 http://localhost:8001/mcp/sse 上运行，可以使用支持SSE的客户端访问。

## 自定义模型训练

如需训练自己的模型:

```bash
cd eth_predict
python train_eth.py
```

训练完成后的模型将保存在 `./eth_predict/saved_models` 目录下。
