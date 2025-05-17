package bitget

import (
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	"github.com/goccy/go-json"
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
	ex, _ := New(exBytes)

	accountBalance, err := ex.MarketOrder("BTCUSDT", base.BID, "0.001")
	if err != nil {
		t.Error(err)
	}
	fmt.Println(accountBalance)
}

func TestGetK(t *testing.T) {
	config.InitConfigPath("/Users/furad/Projects/Mix-MCP/static/config.json")

	exCf, err := config.ReadExchangeConfig("bitget")
	if err != nil {
		t.Error(err)
	}
	exBytes, err := json.Marshal(exCf)
	if err != nil {
		t.Error(err)
	}
	ex, _ := New(exBytes)

	accountBalance, err := ex.GetHistoryCandles("BTCUSDT", "1h", TimesStamp())
	if err != nil {
		t.Error(err)
	}
	fmt.Println(accountBalance)
}
