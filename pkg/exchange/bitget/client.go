package bitget

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/bitly/go-simplejson"
	"github.com/yasseldg/bitget/constants"
)

type Client struct {
	accessKey string
	secretKey string
	password  string
	baseUrl   string
	client    *http.Client
	signer    *Signer
}

func New(params []byte) (c Client, err error) {
	c.client = &http.Client{
		Timeout: time.Duration(30) * time.Second,
	}

	sj, err := simplejson.NewJson(params)
	if err != nil {
		return c, err
	}

	// set init params
	c.baseUrl = sj.Get("url").MustString()
	c.accessKey = sj.Get("apiKey").MustString()
	c.secretKey = sj.Get("secretKey").MustString()
	c.password = sj.Get("password").MustString()
	c.signer = new(Signer).Init(c.secretKey)

	return c, nil
}

func (c *Client) NewWithParams(baseUrl, accessKey, secretKey, password string) error {
	c.client = &http.Client{
		Timeout: time.Duration(30) * time.Second,
	}

	c.baseUrl = baseUrl
	c.accessKey = accessKey
	c.secretKey = secretKey
	c.password = password
	c.signer = new(Signer).Init(c.secretKey)

	return nil
}

func TimesStamp() string {
	timesStamp := time.Now().Unix() * 1000
	return strconv.FormatInt(timesStamp, 10)
}

func TimesStampSec() string {
	timesStamp := time.Now().Unix()
	return strconv.FormatInt(timesStamp, 10)
}

func Headers(request *http.Request, apikey string, timestamp string, sign string, passphrase string) {
	request.Header.Add(constants.ContentType, constants.ApplicationJson)
	request.Header.Add(constants.BgAccessKey, apikey)
	request.Header.Add(constants.BgAccessSign, sign)
	request.Header.Add(constants.BgAccessTimestamp, timestamp)
	request.Header.Add(constants.BgAccessPassphrase, passphrase)
	request.Header.Add("paptrading", strconv.Itoa(1))
}

func NewParams() map[string]string {
	return make(map[string]string)
}

func ToJson(v interface{}) (string, error) {
	result, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(result), nil
}

func BuildJsonParams(params map[string]string) (string, error) {
	if params == nil {
		return "", errors.New("illegal parameter")
	}
	data, err := json.Marshal(params)
	if err != nil {
		return "", errors.New("json convert string error")
	}
	jsonBody := string(data)
	return jsonBody, nil
}

func BuildGetParams(params map[string]string) string {
	urlParams := url.Values{}
	if params != nil && len(params) > 0 {
		for k := range params {
			urlParams.Add(k, params[k])
		}
	}
	return "?" + urlParams.Encode()
}

func (c *Client) DoGet(uri string, params map[string]string) (*http.Response, error) {
	timesStamp := TimesStamp()
	body := BuildGetParams(params)

	sign := c.signer.Sign(constants.GET, uri, body, timesStamp)

	requestUrl := c.baseUrl + uri + body

	request, err := http.NewRequest(constants.GET, requestUrl, nil)
	if err != nil {
		return nil, err
	}
	Headers(request, c.accessKey, timesStamp, sign, c.password)

	response, err := c.client.Do(request)

	if err != nil {
		return nil, err
	}

	return response, err
}

func (c *Client) DoPost(uri string, params map[string]string) (*http.Response, error) {
	timesStamp := TimesStamp()
	//body, _ := internal.BuildJsonParams(params)
	body, _ := BuildJsonParams(params)
	bodyJson, err := json.Marshal(params)
	sign := c.signer.Sign(constants.POST, uri, body, timesStamp)
	requestUrl := c.baseUrl + uri
	buffer := strings.NewReader(string(bodyJson))
	request, err := http.NewRequest(constants.POST, requestUrl, buffer)

	Headers(request, c.accessKey, timesStamp, sign, c.password)
	if err != nil {
		return nil, err
	}
	response, err := c.client.Do(request)

	if err != nil {
		return nil, err
	}

	return response, err
}

func (c *Client) GetAccountBalance(currency string) ([]string, error) {
	params := NewParams()
	if len(currency) > 0 {
		params["coin"] = currency
	}
	uri := "/api/v2/spot/account/assets"

	resp, err := c.DoGet(uri, params)
	var balance struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Data    []struct {
			CoinId    int64  `json:"coinId"`
			CoinName  string `json:"coinName"`
			Available string `json:"available"`
			Frozen    string `json:"frozen"`
			Lock      string `json:"lock"`
			UTime     string `json:"uTime"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return []string{"0", "0", "0"}, fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return []string{"0", "0", "0"}, err
	}

	err = json.Unmarshal(data, &balance)
	if err != nil {
		return []string{"0", "0", "0"}, err
	}
	var balanceList []string
	if balance.Code == "00000" {
		a, _ := strconv.ParseFloat(balance.Data[0].Available, 64)
		l, _ := strconv.ParseFloat(balance.Data[0].Frozen, 64)
		t := strconv.FormatFloat(a+l, 'f', 4, 64)
		return append(balanceList, balance.Data[0].Available, balance.Data[0].Frozen, t), nil
	}
	return []string{"0", "0", "0"}, nil
}

func (c *Client) MarketOrder(symbol, side, size string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["orderType"] = "market"
	params["force"] = "gtc"
	if side == base.BID {
		params["side"] = "buy"
	} else if side == base.ASK {
		params["side"] = "sell"
	}
	if side == base.ASK {
		params["size"] = size
	} else {
		price, err := c.GetMarketPrice(symbol)
		if err != nil {
			return "", err
		}
		p, _ := strconv.ParseFloat(price, 64)
		s, _ := strconv.ParseFloat(size, 64)
		params["size"] = strconv.FormatFloat(s*p, 'f', 2, 64)
	}
	uri := "/api/v2/spot/trade/place-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			OrderId       string `json:"orderId"`
			ClientOrderId string `json:"clientOrderId"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	err = json.Unmarshal(data, &result)
	if err != nil {
		return "", err
	}
	return result.Data.OrderId, err
}

func (c *Client) GetMarketPrice(symbol string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol

	uri := "/api/v2/spot/market/tickers"

	resp, err := c.DoGet(uri, params)

	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data []struct {
			LastPr string `json:"lastPr"`
		} `json:"data"`
	}

	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	err = json.Unmarshal(data, &result)
	if len(result.Data) > 0 {
		return result.Data[0].LastPr, err
	}
	return "", err
}

func (c *Client) Depth(symbol, limit string) (base.WsData, error) {
	params := NewParams()
	params["symbol"] = symbol
	if len(limit) > 0 {
		params["limit"] = limit
	}

	params["type"] = "step0"

	uri := "/api/v2/spot/market/orderbook"

	resp, err := c.DoGet(uri, params)
	if err != nil {
		return base.WsData{}, err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			Asks      [][]string `json:"asks"`
			Bids      [][]string `json:"bids"`
			Timestamp string     `json:"ts"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return base.WsData{}, fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return base.WsData{}, err
	}
	err = json.Unmarshal(data, &result)
	var depth base.WsData
	t, _ := strconv.Atoi(result.Data.Timestamp)
	depth.Time = int64(t)
	var ask, bid []base.PriceLevel
	for _, a := range result.Data.Asks {
		ask = append(ask, base.PriceLevel{
			Price:    a[0],
			Quantity: a[1],
		})
	}
	for _, b := range result.Data.Bids {
		bid = append(bid, base.PriceLevel{
			Price:    b[0],
			Quantity: b[1],
		})
	}
	depth.Asks = ask
	depth.Bids = bid
	return depth, err
}

func (c *Client) GetOrder(symbol, id string) (base.OrderInfo, error) {
	params := NewParams()
	params["orderId"] = id
	uri := "/api/v2/spot/trade/orderInfo"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return base.OrderInfo{}, err
	}
	var result struct {
		Code        string `json:"code"`
		Msg         string `json:"msg"`
		RequestTime int64  `json:"requestTime"`
		Data        []struct {
			UserId           string `json:"userId"`
			Symbol           string `json:"symbol"`
			OrderId          string `json:"orderId"`
			ClientOrderId    string `json:"clientOrderId"`
			Price            string `json:"price"`
			Size             string `json:"size"`
			OrderType        string `json:"orderType"`
			Side             string `json:"side"`
			Status           string `json:"status"`
			FillPrice        string `json:"fillPrice"`
			FillQuantity     string `json:"fillQuantity"`
			FillTotalAmount  string `json:"fillTotalAmount"`
			EnterPointSource string `json:"enterPointSource"`
			FeeDetail        string `json:"feeDetail"`
			OrderSource      string `json:"orderSource"`
			CTime            string `json:"cTime"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return base.OrderInfo{}, fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return base.OrderInfo{}, err
	}

	err = json.Unmarshal(data, &result)
	var typ, status, side string
	if err != nil {
		return base.OrderInfo{}, err
	}
	t, _ := strconv.Atoi(result.Data[0].CTime)
	if result.Data[0].OrderType == "limit" {
		typ = base.LIMIT
	} else if result.Data[0].OrderType == "market" {
		typ = base.MARKET
	}
	if result.Data[0].Side == "buy" {
		side = base.BID
	} else if result.Data[0].Side == "sell" {
		side = base.ASK
	}
	if result.Data[0].Status == "new" {
		status = base.OPEN
	} else if result.Data[0].Status == "partial_fill" {
		status = base.PARTIALLY
	} else if result.Data[0].Status == "full_fill" {
		status = base.FILLED
	} else if result.Data[0].Status == "cancelled" {
		status = base.CANCELED
	}
	info := base.OrderInfo{
		OrderID:  result.Data[0].OrderId,
		Symbol:   symbol,
		Side:     side,
		Price:    result.Data[0].Price,
		Quantity: result.Data[0].Size,
		Type:     typ,
		Filled:   result.Data[0].FillQuantity,
		USDT:     result.Data[0].FillTotalAmount,
		Status:   status,
		Time:     int64(t),
	}
	return info, err

}

func (c *Client) LimitOrder(symbol, side, price, size string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["orderType"] = "limit"
	params["force"] = "gtc"
	if side == base.BID {
		params["side"] = "buy"
	} else if side == base.ASK {
		params["side"] = "sell"
	}

	params["size"] = size
	params["price"] = price
	uri := "/api/v2/spot/trade/place-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			OrderId       string `json:"orderId"`
			ClientOrderId string `json:"clientOrderId"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	//fmt.Println(string(data))
	err = json.Unmarshal(data, &result)
	if err != nil {
		return "", err
	}
	return result.Data.OrderId, err
}

func (c *Client) MakerOrder(symbol, side, price, size string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["orderType"] = "limit"
	params["force"] = "post_only"
	if side == base.BID {
		params["side"] = "buy"
	} else if side == base.ASK {
		params["side"] = "sell"
	}
	params["size"] = size
	params["price"] = price
	uri := "/api/v2/spot/trade/place-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			OrderId       string `json:"orderId"`
			ClientOrderId string `json:"clientOrderId"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	//fmt.Println(string(data))
	err = json.Unmarshal(data, &result)
	if err != nil {
		return "", err
	}
	return result.Data.OrderId, err
}

func (c *Client) TakerOrder(symbol, side, price, size string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["orderType"] = "limit"
	params["force"] = "ioc"
	if side == base.BID {
		params["side"] = "buy"
	} else if side == base.ASK {
		params["side"] = "sell"
	}

	params["size"] = size
	params["price"] = price
	uri := "/api/v2/spot/trade/place-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			OrderId       string `json:"orderId"`
			ClientOrderId string `json:"clientOrderId"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	//fmt.Println(string(data))
	err = json.Unmarshal(data, &result)
	if err != nil {
		return "", err

	}
	return result.Data.OrderId, err
}

func (c *Client) CancelOrder(symbol, id string) (bool, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["orderId"] = id
	uri := "/api/v2/spot/trade/cancel-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return false, err
	}
	var result struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Data    string `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return false, fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, err
	}

	err = json.Unmarshal(data, &result)
	if err != nil && result.Message == "success" {
		return false, err
	}
	return true, err
}

func (c *Client) CancelOrders(symbol string) error {
	params := NewParams()
	params["symbol"] = symbol

	uri := "/api/v2/spot/trade/cancel-symbol-order"
	resp, err := c.DoPost(uri, params)
	if err != nil {
		return err
	}
	var result struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Data    string `json:"data"`
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		fmt.Println(data)
		return err
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	err = json.Unmarshal(data, &result)
	if err != nil && result.Message == "success" {
		return err
	}
	return nil
}

func (c *Client) GetHistoryCandles(symbol string, granularity string, endTime string) ([][]string, error) {
	params := NewParams()
	params["symbol"] = symbol
	params["granularity"] = granularity
	params["endTime"] = endTime
	params["limit"] = "6"

	uri := "/api/v2/spot/market/history-candles"

	resp, err := c.DoGet(uri, params)

	if err != nil {
		return nil, err
	}
	var result struct {
		Code        string     `json:"code"`
		Msg         string     `json:"msg"`
		RequestTime int64      `json:"requestTime"`
		Data        [][]string `json:"data"`
	}

	if resp.StatusCode != http.StatusOK {
		data, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("response status code is not OK, response code is %d, body:%s", resp.StatusCode, string(data))
	}

	if resp != nil && resp.Body != nil {
		defer resp.Body.Close()
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &result)
	if len(result.Data) > 0 {
		return result.Data, err
	}
	return nil, err
}
