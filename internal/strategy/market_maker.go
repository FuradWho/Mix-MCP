package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type MakerCfg struct {
	Symbol string
	Size   string          // 每张委托数量
	Spread decimal.Decimal // %，如 0.002 == 0.2%
	Tick   time.Duration
}

func PassiveMaker(ctx context.Context, ex store.ExchangeStore, cfg MakerCfg) error {
	var bidID, askID string
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			break
		}

		priceStr, err := ex.GetMarketPrice(cfg.Symbol)
		if err != nil {
			log.Println("price err:", err)
			continue
		}
		mid, _ := decimal.NewFromString(priceStr)

		// 2. 目标价
		half := mid.Mul(cfg.Spread).Div(decimal.NewFromInt(2))
		bid := mid.Sub(half)
		ask := mid.Add(half)

		// 3. 更新/下单
		if bidID != "" {
			ex.CancelOrder(cfg.Symbol, bidID)
		}
		if askID != "" {
			ex.CancelOrder(cfg.Symbol, askID)
		}

		if bidID, err = ex.MakerOrder(cfg.Symbol, "buy", bid.String(), cfg.Size); err != nil {
			log.Println("maker buy err:", err)
		}
		if askID, err = ex.MakerOrder(cfg.Symbol, "sell", ask.String(), cfg.Size); err != nil {
			log.Println("maker sell err:", err)
		}

		time.Sleep(cfg.Tick)
	}
}
