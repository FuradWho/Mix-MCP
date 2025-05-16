package base

const (
	BID       = "bid"
	ASK       = "ask"
	OPEN      = "open"
	FILLED    = "filled"
	CANCELED  = "canceled"
	PARTIALLY = "partially"

	LIMIT        = "limit"
	LIMITHIDDEN  = "limit_hidden"
	MAKER        = "maker"
	TAKER        = "taker"
	MARKET       = "market"
	OPEN_ORDERS  = "open_orders"
	LAYER_ORDERS = "layer_orders"
)

type WsData struct {
	Time int64        `json:"time"`
	Bids []PriceLevel `json:"bids"`
	Asks []PriceLevel `json:"asks"`
}

type PriceLevel struct {
	Price    string `json:"price"`
	Quantity string `json:"quantity"`
}

type PositionInfo struct {
	Symbol           string `json:"symbol"`
	PositionAmt      string `json:"positionAmt"`
	EntryPrice       string `json:"entryPrice"`
	MarkPrice        string `json:"markPrice"`
	UnRealizedProfit string `json:"unRealizedProfit"`
	LiquidationPrice string `json:"liquidationPrice"`
	Leverage         string `json:"leverage"`
	MaxNotionalValue string `json:"maxNotionalValue"`
	MarginType       string `json:"marginType"`
	IsolatedMargin   string `json:"isolatedMargin"`
	IsAutoAddMargin  string `json:"isAutoAddMargin"`
	PositionSide     string `json:"positionSide"`
	Notional         string `json:"notional"`
	IsolatedWallet   string `json:"isolatedWallet"`
	UpdateTime       int64  `json:"updateTime"`
}

type OrderInfo struct {
	OrderID string `json:"order_id"`
	Symbol  string `json:"symbol"`
	Side    string `json:"side"`
	Price   string `json:"price"`
	// Size     string `json:"size"`
	Quantity string `json:"quantity"`
	Type     string `json:"type"`
	Filled   string `json:"filled"`
	USDT     string `json:"usdt"`
	Status   string `json:"status"`
	Time     int64  `json:"time"`
}

type Balance struct {
	Asset  string `json:"asset"`
	Free   string `json:"free"`
	Locked string `json:"locked"`
}
