package bitget

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/base"
	"github.com/yasseldg/bitget/config"
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

func (c *Client) New(params []byte) error {
	c.client = &http.Client{
		Timeout: time.Duration(30) * time.Second,
	}

	sj, err := simplejson.NewJson(params)
	if err != nil {
		return err
	}

	// set init params
	c.baseUrl = sj.Get("url").MustString()
	c.accessKey = sj.Get("apiKey").MustString()
	c.secretKey = sj.Get("secretKey").MustString()
	c.password = sj.Get("password").MustString()
	c.signer = new(Signer).Init(c.secretKey)

	return nil
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
	requestUrl := config.BaseUrl + uri
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

func (c *Client) MarketOrder(symbol, side, size string) (string, error) {
	params := NewParams()
	params["symbol"] = symbol + "_SPBL"
	params["orderType"] = "market"
	params["force"] = "normal"
	if side == base.BID {
		params["side"] = "buy"
	} else if side == base.ASK {
		params["side"] = "sell"
	}
	if side == base.ASK {
		params["quantity"] = size
	} else {
		price, err := c.GetMarketPrice(symbol)
		if err != nil {
			return "", err
		}
		p, _ := strconv.ParseFloat(price, 64)
		s, _ := strconv.ParseFloat(size, 64)
		params["quantity"] = strconv.FormatFloat(s*p, 'f', 2, 64)
	}
	uri := constants.SpotTrade + "/orders"
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
	params["symbol"] = symbol + "_SPBL"

	uri := constants.SpotMarket + "/ticker"

	resp, err := c.DoGet(uri, params)

	if err != nil {
		return "", err
	}
	var result struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
		Data struct {
			Symbol    string `json:"symbol"`
			High24H   string `json:"high24h"`
			Low24H    string `json:"low24h"`
			Close     string `json:"close"`
			QuoteVol  string `json:"quoteVol"`
			BaseVol   string `json:"baseVol"`
			UsdtVol   string `json:"usdtVol"`
			Ts        string `json:"ts"`
			BuyOne    string `json:"buyOne"`
			SellOne   string `json:"sellOne"`
			BidSz     string `json:"bidSz"`
			AskSz     string `json:"askSz"`
			OpenUtc0  string `json:"openUtc0"`
			ChangeUtc string `json:"changeUtc"`
			Change    string `json:"change"`
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
	return result.Data.Close, err
}
