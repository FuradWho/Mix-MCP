package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type MakerArgs struct {
	Symbol string  `json:"symbol"  jsonschema:"required,description=交易对符号，例如 BTC-USDT"`
	Size   string  `json:"size"    jsonschema:"required,description=单边挂单数量"`
	Spread float64 `json:"spread"  jsonschema:"description=总价差百分比，如 0.002 表示 0.2%"`
	Tick   int64   `json:"tick"    jsonschema:"description=刷新报价间隔（秒）"`
}

func PassiveMaker(ctx context.Context, ex store.ExchangeStore, cfg MakerArgs) error {
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
		half := mid.Mul(decimal.NewFromFloat(cfg.Spread)).Div(decimal.NewFromInt(2))
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

		time.Sleep(time.Duration(cfg.Tick))
	}
}
