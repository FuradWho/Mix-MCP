// breakout.go
package strategy

import (
	"context"
	"github.com/FuradWho/Mix-MCP/internal/store"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

type BreakoutCfg struct {
	Symbol   string
	Size     string
	Lookback int
	Buffer   decimal.Decimal // 跳过点
	TP       decimal.Decimal // 止盈 %
	SL       decimal.Decimal // 止损 %
	Tick     time.Duration
}

func Breakout(ctx context.Context, ex store.ExchangeStore, cfg BreakoutCfg) error {
	var highs []decimal.Decimal

	for {
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

		trigger := maxHigh.Mul(decimal.NewFromInt(1).Add(cfg.Buffer))
		if p.GreaterThan(trigger) {
			// 1. 追涨
			orderID, err := ex.MarketOrder(cfg.Symbol, "buy", cfg.Size)
			if err != nil {
				log.Println("mkt buy:", err)
				continue
			}
			log.Println("breakout buy id", orderID)

			entry := p
			tp := entry.Mul(decimal.NewFromInt(1).Add(cfg.TP))
			sl := entry.Mul(decimal.NewFromInt(1).Sub(cfg.SL))

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
					ex.MarketOrder(cfg.Symbol, "sell", cfg.Size)
					log.Println("exit at", px)
					break
				}
				time.Sleep(cfg.Tick)
			}
			highs = nil // 重新统计
		}
		time.Sleep(cfg.Tick)
	}
}

func fetch(p string, e error) string { return p } // 简化处理
