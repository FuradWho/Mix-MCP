// dca.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type DCAArgs struct {
	Symbol   string  `json:"symbol"     jsonschema:"required,description=交易对符号"`
	FiatSize float64 `json:"fiat_size"  jsonschema:"required,description=每期投入的法币金额"`
	Interval int64   `json:"interval"   jsonschema:"required,description=两次买入之间的间隔（秒）"`
	Times    int     `json:"times"      jsonschema:"required,description=执行总次数"`
}

func DCA(ctx context.Context, ex store.ExchangeStore, cfg DCAArgs) error {
	for i := 0; i < cfg.Times; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			break
		}

		pStr, err := ex.GetMarketPrice(cfg.Symbol)
		if err != nil {
			log.Println(err)
			continue
		}
		price, _ := decimal.NewFromString(pStr)

		coinAmt := decimal.NewFromFloat(cfg.FiatSize).Div(price)
		_, err = ex.MarketOrder(cfg.Symbol, "buy", coinAmt.String())
		if err != nil {
			log.Println("dca buy:", err)
		}

		time.Sleep(time.Duration(cfg.Interval))
	}
	return nil
}
