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
