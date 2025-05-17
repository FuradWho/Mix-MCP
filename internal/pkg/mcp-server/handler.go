package mcpserver

import (
	"context"
	"encoding/json"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/internal/strategy"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
	"log"
	"strconv"
	"time"
)

type GridArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	strategy.GridArgs
}

type DCAArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	strategy.DCAArgs
}

type MakerArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	strategy.MakerArgs
}

type CancelStrategyArgs struct {
	StrategyName string `json:"strategy_name"  jsonschema:"required,description=策略名"`
}

type HistoryCandleArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	Symbol       string `json:"symbol"  jsonschema:"required,description=币对名, eg.BTCUSDT"`
	Granularity  string `json:"granularity" jsonschema:"required,description=K线的时间间隔 分钟：1min，5min，15min，30min 小时：1h，4h，6h，12h天：1day，3day周：1week月：1M零时区小时线：6Hutc，12Hutc零时区日线：1Dutc ，3Dutc零时区周线：1Wutc零时区月线：1Mutc"`
}

type CurrentPriceArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	Symbol       string `json:"symbol"  jsonschema:"required,description=币对名, eg.BTCUSDT"`
}

type DateArgs struct {
	TimesStamp int64 `json:"times_stamp"  jsonschema:"required,description=时间戳"`
}

type NullArgs struct {
}

func GridHandler(args GridArgs) (*mcp.ToolResponse, error) {
	log.Println("grid strategy")
	args.Symbol = strategy.FormatSymbol(args.ExchangeName, args.Symbol)
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		err = strategy.PushStrategyCtx("grid", cancel)
		if err != nil {
			log.Println(err.Error())
			return
		}
		err = strategy.Grid(ctx, ex, args.GridArgs)
		if err != nil {
			strategy.PopStrategyCtx("grid")
			log.Println(err.Error())
			return
		}
	}()
	time.Sleep(time.Second * 3)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}

	return mcp.NewToolResponse(mcp.NewTextContent("success")), err
}

func DCAHandler(args DCAArgs) (*mcp.ToolResponse, error) {
	args.Symbol = strategy.FormatSymbol(args.ExchangeName, args.Symbol)
	log.Println("dca strategy", args.ExchangeName, args.DCAArgs.Symbol)
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		err = strategy.PushStrategyCtx("dca", cancel)
		if err != nil {
			log.Println(err.Error())
			return
		}
		err = strategy.DCA(ctx, ex, args.DCAArgs)
		if err != nil {
			strategy.PopStrategyCtx("dca")
			log.Println(err.Error())
			return
		}
	}()
	time.Sleep(time.Second * 3)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}

	return mcp.NewToolResponse(mcp.NewTextContent("success")), err
}

func MakerHandler(args MakerArgs) (*mcp.ToolResponse, error) {
	log.Println("maker strategy")
	args.Symbol = strategy.FormatSymbol(args.ExchangeName, args.Symbol)
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		err = strategy.PushStrategyCtx("maker", cancel)
		if err != nil {
			log.Println(err.Error())
			return
		}
		err = strategy.PassiveMaker(ctx, ex, args.MakerArgs)
		if err != nil {
			strategy.PopStrategyCtx("maker")
			log.Println(err.Error())
			return
		}
	}()
	time.Sleep(time.Second * 3)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}

	return mcp.NewToolResponse(mcp.NewTextContent("success")), err
}

func CancelStrategy(args CancelStrategyArgs) (*mcp.ToolResponse, error) {
	strategy.PopStrategyCtx(args.StrategyName)
	return mcp.NewToolResponse(mcp.NewTextContent("success")), nil
}

func HistoryCandleHandler(args HistoryCandleArgs) (*mcp.ToolResponse, error) {
	args.Symbol = strategy.FormatSymbol(args.ExchangeName, args.Symbol)
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	log.Println("history candle", args.ExchangeName, args.Granularity, args.Symbol)
	candleData, err := ex.GetHistoryCandles(args.Symbol, args.Granularity, strconv.FormatInt(time.Now().Unix()*1000, 10))
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	candleBytes, err := json.Marshal(candleData)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	return mcp.NewToolResponse(mcp.NewTextContent(string(candleBytes))), err
}

func CurrentPriceHandler(args CurrentPriceArgs) (*mcp.ToolResponse, error) {
	args.Symbol = strategy.FormatSymbol(args.ExchangeName, args.Symbol)
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	price, err := ex.GetMarketPrice(args.Symbol)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	return mcp.NewToolResponse(mcp.NewTextContent(price)), err
}

func GetStrategyInfoHandler(null NullArgs) (*mcp.ToolResponse, error) {

	info := `{
  "strategies": [
    {
      "name": "PassiveMaker",
      "description": "被动做市：在最新中间价上下固定价差同时挂买单与卖单，赚取点差。",
      "parameters": {
        "symbol":   { "type": "string",  "required": true,  "description": "交易对符号，例如 BTC-USDT" },
        "size":     { "type": "string",  "required": true,  "description": "每张挂单的交易数量" },
        "spread":   { "type": "float64", "required": false, "description": "总价差百分比，如 0.002 表示 0.2%" },
        "tick":     { "type": "int64",   "required": false, "description": "刷新并重挂委托的时间间隔（秒）" }
      }
    },
    {
      "name": "BreakoutMomentum",
      "description": "突破动量：价格突破近期最高价后市价追单，配套止盈止损。",
      "parameters": {
        "symbol":   { "type": "string",  "required": true,  "description": "交易对符号" },
        "size":     { "type": "string",  "required": true,  "description": "市价开仓数量" },
        "lookback": { "type": "int",     "required": true,  "description": "统计最高价的回看 K 线数量" },
        "buffer":   { "type": "float64", "required": true,  "description": "突破缓冲百分比，避免假突破" },
        "tp":       { "type": "float64", "required": true,  "description": "止盈百分比" },
        "sl":       { "type": "float64", "required": true,  "description": "止损百分比" },
        "tick":     { "type": "int64",   "required": false, "description": "检查平仓条件的轮询间隔（秒）" }
      }
    },
    {
      "name": "GridTrading",
      "description": "网格均值回归：在预设价格区间内划分若干网格，价格下跌买入、上涨卖出，利用震荡获利。",
      "parameters": {
        "symbol": { "type": "string",  "required": true, "description": "交易对符号" },
        "low":    { "type": "float64", "required": true, "description": "网格下边界价格" },
        "high":   { "type": "float64", "required": true, "description": "网格上边界价格" },
        "levels": { "type": "int",     "required": true, "description": "网格层数（间隔数量）" },
        "size":   { "type": "string",  "required": true, "description": "每格委托数量" }
      }
    },
    {
      "name": "DollarCostAveraging",
      "description": "定投：按固定时间间隔用固定法币金额市价买入，摊平成本。",
      "parameters": {
        "symbol":    { "type": "string",  "required": true, "description": "交易对符号" },
        "fiat_size": { "type": "float64", "required": true, "description": "每期投入的法币金额（例如 USDT）" },
        "interval":  { "type": "int64",   "required": true, "description": "两次买入间的时间间隔（秒）" },
        "times":     { "type": "int",     "required": true, "description": "总执行次数" }
      }
    }
  ]
}`
	return mcp.NewToolResponse(mcp.NewTextContent(info)), nil
}

func GetDate(t DateArgs) (*mcp.ToolResponse, error) {
	tm := time.Unix(t.TimesStamp/1000, 0)
	return mcp.NewToolResponse(mcp.NewTextContent(tm.Format("2006-01-02 15:04:05"))), nil
}
