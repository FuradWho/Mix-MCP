package bitget

import (
	"net/http"
	"time"

	"github.com/bitly/go-simplejson"
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
