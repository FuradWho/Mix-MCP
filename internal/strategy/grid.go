// grid.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"

	"github.com/shopspring/decimal"
)

type GridCfg struct {
	Symbol    string
	Low, High decimal.Decimal
	Levels    int
	Size      string
}

func Grid(ctx context.Context, ex store.ExchangeStore, cfg GridCfg) error {
	step := cfg.High.Sub(cfg.Low).Div(decimal.NewFromInt(int64(cfg.Levels)))
	var prices []decimal.Decimal
	for i := 0; i <= cfg.Levels; i++ {
		prices = append(prices, cfg.Low.Add(step.Mul(decimal.NewFromInt(int64(i)))))
	}

	ids := make(map[string]string) // price -> id
	for _, p := range prices {
		side := "buy"
		if p.GreaterThan(cfg.Low.Add(cfg.High).Div(decimal.NewFromInt(2))) {
			side = "sell"
		}
		id, err := ex.LimitOrder(cfg.Symbol, side, p.String(), cfg.Size)
		if err != nil {
			log.Println(err)
			continue
		}
		ids[p.String()] = id
	}

	// 实际使用时：监听成交事件并在相邻价格补单，可配合 WebSocket
	<-ctx.Done()
	return ctx.Err()
}
