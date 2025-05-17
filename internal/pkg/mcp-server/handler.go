package mcpserver

import (
	"context"
	"encoding/json"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/internal/strategy"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
)

type GridArgs struct {
	ExchangeName string `json:"exchange_name"  jsonschema:"required,description=交易所名"`
	strategy.GridArgs
}

func GridHandler(args GridArgs) (*mcp.ToolResponse, error) {
	exCf, err := config.ReadExchangeConfig(args.ExchangeName)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	ex := store.NewStore().Bitget(exBytes)
	err = strategy.Grid(context.Background(), ex, args.GridArgs)
	if err != nil {
		return mcp.NewToolResponse(mcp.NewTextContent(err.Error())), err
	}
	return mcp.NewToolResponse(mcp.NewTextContent("success")), err
}
