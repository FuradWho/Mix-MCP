package mcpserver

import (
	"context"
	"encoding/json"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/internal/strategy"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
	"log"
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

func GridHandler(args GridArgs) (*mcp.ToolResponse, error) {
	log.Println("grid strategy")
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
	log.Println("dca strategy")
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
