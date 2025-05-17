// grid.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"log"

	"github.com/shopspring/decimal"
)

type GridArgs struct {
	Symbol string  `json:"symbol"  jsonschema:"required,description=交易对符号"`
	Low    float64 `json:"low"     jsonschema:"required,description=网格下边界价格"`
	High   float64 `json:"high"    jsonschema:"required,description=网格上边界价格"`
	Levels int     `json:"levels"  jsonschema:"required,description=网格层数"`
	Size   string  `json:"size"    jsonschema:"required,description=每格委托数量"`
}

func Grid(ctx context.Context, ex store.ExchangeStore, cfg GridArgs) error {
	step := decimal.NewFromFloat(cfg.High).Sub(decimal.NewFromFloat(cfg.Low)).Div(decimal.NewFromInt(int64(cfg.Levels)))
	var prices []decimal.Decimal
	for i := 0; i <= cfg.Levels; i++ {
		prices = append(prices, decimal.NewFromFloat(cfg.Low).Add(step.Mul(decimal.NewFromInt(int64(i)))))
	}

	ids := make(map[string]string) // price -> id
	for _, p := range prices {
		side := base.BID
		if p.GreaterThan(decimal.NewFromFloat(cfg.Low).Add(decimal.NewFromFloat(cfg.High)).Div(decimal.NewFromInt(2))) {
			side = base.ASK
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
	ex.CancelOrders(cfg.Symbol)
	return ctx.Err()
}
