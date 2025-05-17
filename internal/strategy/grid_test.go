package strategy

import (
	"context"
	"encoding/json"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	"testing"
)

func TestGrid(t *testing.T) {
	config.InitConfigPath("/Users/furad/Projects/Mix-MCP/static/config.json")

	exCf, err := config.ReadExchangeConfig("bitget")
	if err != nil {
		t.Error(err)
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		t.Error(err)
	}
	ex := store.NewStore().Bitget(exBytes)

	err = Grid(context.Background(), ex, GridArgs{
		Symbol: "BTCUSDT",
		Low:    70000,
		High:   140000,
		Levels: 20,
		Size:   "0.01",
	})
	if err != nil {
		t.Error(err)
	}
}
