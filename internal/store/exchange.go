package store

import (
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"github.com/FuradWho/Mix-MCP/pkg/exchange/bitget"
)

type IStore interface {
	Bitget() ExchangeStore
}

type Store struct {
}

func NewStore() *Store {
	return &Store{}
}

var _ ExchangeStore = (*bitget.Client)(nil)

func (s *Store) Bitget(params []byte) ExchangeStore {
	fmt.Println(string(params))
	client, err := bitget.New(params)
	if err != nil {
		return nil
	}

	return &client
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
	GetHistoryCandles(symbol string, granularity string, endTime string) ([][]string, error)
}
