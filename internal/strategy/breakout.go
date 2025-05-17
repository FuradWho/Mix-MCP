// breakout.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type BreakoutArgs struct {
	Symbol   string  `json:"symbol"   jsonschema:"required,description=交易对符号"`
	Size     string  `json:"size"     jsonschema:"required,description=市价开仓数量"`
	Lookback int     `json:"lookback" jsonschema:"required,description=回看 N 根 K 线的最高价"`
	Buffer   float64 `json:"buffer"   jsonschema:"required,description=突破缓冲百分比"`
	TP       float64 `json:"tp"       jsonschema:"required,description=止盈百分比"`
	SL       float64 `json:"sl"       jsonschema:"required,description=止损百分比"`
	Tick     int64   `json:"tick"     jsonschema:"description=轮询价格间隔（秒）"`
}

func Breakout(ctx context.Context, ex store.ExchangeStore, cfg BreakoutArgs) error {
	var highs []decimal.Decimal

	for {
		select {
		case <-ctx.Done():
			ex.CancelOrders(cfg.Symbol)
			return ctx.Err()
		default:
			break
		}

		pStr, err := ex.GetMarketPrice(cfg.Symbol)
		if err != nil {
			log.Println(err)
			continue
		}
		p, _ := decimal.NewFromString(pStr)

		// 记录最高价，维护长度 = Lookback
		highs = append(highs, p)
		if len(highs) > cfg.Lookback {
			highs = highs[1:]
		}

		// 判断突破
		maxHigh := highs[0]
		for _, v := range highs {
			if v.GreaterThan(maxHigh) {
				maxHigh = v
			}
		}

		trigger := maxHigh.Mul(decimal.NewFromInt(1).Add(decimal.NewFromFloat(cfg.Buffer)))
		if p.GreaterThan(trigger) {
			// 1. 追涨
			orderID, err := ex.MarketOrder(cfg.Symbol, base.BID, cfg.Size)
			if err != nil {
				log.Println("mkt buy:", err)
				continue
			}
			log.Println("breakout buy id", orderID)

			entry := p
			tp := entry.Mul(decimal.NewFromInt(1).Add(decimal.NewFromFloat(cfg.TP)))
			sl := entry.Mul(decimal.NewFromInt(1).Sub(decimal.NewFromFloat(cfg.SL)))

			// 2. 止盈/止损循环
			for {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					break
				}
				px, _ := decimal.NewFromString(fetch(ex.GetMarketPrice(cfg.Symbol)))
				if px.GreaterThan(tp) || px.LessThan(sl) {
					ex.MarketOrder(cfg.Symbol, base.BID, cfg.Size)
					log.Println("exit at", px)
					break
				}
				time.Sleep(time.Duration(cfg.Tick))
			}
			highs = nil // 重新统计
		}
		time.Sleep(time.Duration(cfg.Tick))
	}
}

func fetch(p string, e error) string { return p } // 简化处理
