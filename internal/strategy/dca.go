// dca.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type DCACfg struct {
	Symbol   string          // 如 BTC-USDT
	FiatSize decimal.Decimal // 每期花多少钱 (USDT)
	Interval time.Duration
	Times    int
}

func DCA(ctx context.Context, ex store.ExchangeStore, cfg DCACfg) error {
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

		coinAmt := cfg.FiatSize.Div(price)
		_, err = ex.MarketOrder(cfg.Symbol, "buy", coinAmt.String())
		if err != nil {
			log.Println("dca buy:", err)
		}

		time.Sleep(cfg.Interval)
	}
	return nil
}
