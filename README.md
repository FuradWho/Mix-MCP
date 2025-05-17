<div align="center">
<img src="./static/logo.jpg" height="300" alt="Statusphere logo">
</div>
<br/>
<div align="center">

  ![Language](https://img.shields.io/badge/language-golang-brightgreen)
  ![Language](https://img.shields.io/badge/language-python-brightgreen)
  ![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen)
  ![License](https://img.shields.io/badge/license-MIT-yellow)

</div>

# Mix-MCP

<div align="center" style="font-size: 1.5em;">
  <p><strong><a href="README.md">English</a>, <a href="README_CN.md">中文</a></strong></p>

  <p><strong><a href="https://ethbeijing.gitbook.io/mixmcp">GitBook</a></strong></p>

<a href="https://www.youtube.com/watch?v=wLVySOC8p2k" style="display: inline-block; width: 45%; text-align: left; padding-left: 10px;">
    <img src="https://img.shields.io/badge/Demo-YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="Youtube Demo" style="transform: scale(1.2);">
  </a>
</div>

🌐 Mix-MCP is an innovative platform that bridges the gap between the Web3 World and Large Language Model (LLMs) by leveraging the Model Context Protocol (MCP), which was introduced by Anthropic in 🗓️ December 2024. Our project aims to provide an open platform for any Web3 technology, application, or tool to interact with large language models. 🤖 Any AI+Web3 project can easily interact with large models using our tools!

🔍 Taking the ETH trading scenario as an example, you can engage in conversation with the AI assistant to have it fetch real-time 📊 candlestick data and expert analysis, analyze market indicators, generate market assessment reports, predict ETH trends, construct trading strategies, integrate real-time market assessments into trading strategies, and intelligently initiate or terminate trades📈📉. 
## Highlights
- 🔄 MCP Aggregation: This involves integrating current MCP services for blockchain, such as retrieving on-chain data, and organizing them into a unified, encapsulated interface. This feature aims to simplify data access for developers and traders, reducing the fragmentation seen in multi-chain ecosystems.
- 📈 Market Information Analysis: Based on MCP services, the project will conduct market analysis, with a focus on transaction-related market trends, to provide actionable insights for trading decisions. This leverages AI to process real-time data, potentially offering competitive advantages in volatile markets.
- 🚀 Trading Strategy Implementation: The project includes built-in trading strategies that use the latest MCP data and AI analysis to recommend, optimize, and deploy strategies with one click. This feature targets both novice and professional traders, lowering technical barriers and enhancing automation.

## Advantages
- Practicality of MCP Aggregation: By providing a unified interface for multi-chain data access, MixMCP can significantly lower the usage threshold for developers and traders. This is crucial for scenarios like DeFi and NFT markets, where quick access to on-chain data is essential. The aggregation also supports multi-chain compatibility, a key highlight in today’s fragmented blockchain landscape.
- Commercial Value of Market Analysis: AI-driven analysis of MCP data, especially for trading trends, can offer real-time, precise market insights. This is particularly valuable in the volatile cryptocurrency market, where timely decisions can impact outcomes. Combining on-chain data (e.g., trading volume, holding distribution) with off-chain data (e.g., social media sentiment, macroeconomic indicators) could enhance analysis depth, attracting professional traders and institutions.
- Trading Strategy Automation: Built-in strategies with AI optimization cater to a wide user base, from beginners to experts. The one-click deployment feature reduces technical barriers, making it accessible for users seeking quick implementation. Dynamic adjustments based on market volatility or user risk preferences could further improve user experience and strategy returns.
- Alignment with Trends: The combination of Web3’s decentralization and transparency with AI’s predictive capabilities aligns with the growing blockchain + AI trend, potentially attracting investors and users focused on cutting-edge technology. This synergy has broad applications in algorithmic trading, DeFi yield optimization, and on-chain asset management.

## Getting Started

### 1.Clone The Project
```shell
git clone https://github.com/FuradWho/Mix-MCP
cd Mix-MCP
```
### 2.Install Dependencies
```shell
go get .
```
### 3.Set Up MCP Servers
MixMCP uses multiple MCP-compatible backend services. You can configure them in config.json.
```json
{
  "mcp-servers": [
    {
      "type": "stdio",
      "command": "node",
      "args": [
        "/path/to/index.js"
      ],
      "env": [
        "ALCHEMY_API_KEY=your-alchemy-key"
      ]
    },
    {
      "type": "stdio",
      "command": "node",
      "args": [
        "path/to/index.js"
      ],
      "env": []
    }
  ]
}
```
### 4.Configure Exchange Access
> ⚠️ Important: Keep your API keys secure. Do not commit them to version control.
```json
{
  "exchanges": {
    "bitget": {
      "url": "https://api.bitget.com",
      "apiKey": "your-api-key",
      "secretKey": "your-secret-key",
      "password": "your-api-password"
    },
    "binance": {
      "url": "https://api.binance.com",
      "apiKey": "your-api-key",
      "secretKey": "your-secret-key",
      "password": ""
    }
  }
}
```
### 5.Build
```shell
cd Mix-MCP
go build . 
```
### 6.Use In MCP Client
```json
{
  "mcpServers": {
    "fetch": {
      "command": "path/to/executable",
      "args": [
        "--config",
        "path/to/config.json"
      ]
    }
  }
}
```

## Roadmap
### Phase 1 - Version Alpha
- 50+ MCP Integrated
- Market Analysis Based on MCP Aggregator
- Bitget Trading API Integrated
- Built-in 3+ Trading Strategies
- Automated Trading Strategy Execution
### Phase 2 - Trial & Feedback Collection
- Eliminate Malicious Errors
- Analyze User Experience
- Analyze Trading Strategy Performance
### Phase 3 - Version Beta
- 100+ MCP Integrated
- Solana, EVMs, Sui DEX Trading Integrated
- 20+ CEXs Trading API Integrated
- Built-in 10+ Trading Strategies
- Provides Templates for Custom MCP and Strategies
- Support Historical Data Backtesting
- Visualized Tools for Market Analysis and Trading Management 

## Terms of Use &  Privacy Policy
https://ethbeijing.gitbook.io/mixmcp/terms-of-use-and-privacy-policy
