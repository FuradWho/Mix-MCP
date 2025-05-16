# MCP 本地文件服务

这是一个使用MCP（大模型通信协议）的本地文件服务，它可以通过SSE（Server-Sent Events）格式提供本地文件内容。

## 功能特点

- 读取指定目录下的所有文件内容
- 使用SSE格式实时传输数据
- 支持UTF-8编码的文件读取
- 提供错误处理机制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保在项目根目录下创建 `yuyao` 文件夹
2. 将需要读取的文件放入 `yuyao` 文件夹中
3. 运行服务：

```bash
python main.py
```

4. 访问服务：
   - 服务将在 http://localhost:8000/mcp 上运行
   - 使用支持SSE的客户端（如浏览器或专门的SSE客户端）访问该地址

## 数据格式

服务将以SSE格式返回数据，每个文件的内容将以以下格式发送：

```json
{
    "file_path": "相对路径/文件名",
    "content": "文件内容"
}
```

如果发生错误，将返回错误信息：

```json
{
    "error": "错误信息"
}
``` 