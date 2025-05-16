package store

import (
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"github.com/FuradWho/Mix-MCP/pkg/exchange/bitget"
)

type IStore interface {
	Bitget() ExchangeStore
}

type store struct {
}

func NewStore() *store {
	return &store{}
}

func (s *store) Bitget(params []byte) ExchangeStore {
	client, err := bitget.New(params)
	if err != nil {
		return nil
	}

	return client
}

type ExchangeStore interface {
	GetAccountBalance(currency string) ([]string, error)
	MarketOrder(symbol, side, size string) (string, error)
	LimitOrder(symbol, side, price, size string) (string, error)
	MakerOrder(symbol, side, price, size string) (string, error)
	TakerOrder(symbol, side, price, size string) (string, error)
	CancelOrder(symbol, id string) (bool, error)
	CancelOrders(symbol string) error
	GetOrder(symbol, id string) (base.OrderInfo, error)
	GetMarketPrice(symbol string) (string, error)
	Depth(symbol, limit string) (base.WsData, error)
}
