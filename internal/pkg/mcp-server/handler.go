package mcpserver

import (
	"context"
	"encoding/json"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/internal/strategy"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
	"log"
)

type GridArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	strategy.GridArgs
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
	err = strategy.Grid(context.Background(), ex, args.GridArgs)
	if err != nil {
		log.Println(err.Error())
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	return mcp.NewToolResponse(mcp.NewTextContent("success")), err
}
