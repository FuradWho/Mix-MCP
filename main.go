package main

import (
	"fmt"
	mcpclient "github.com/FuradWho/Mix-MCP/internal/pkg/mcp-client"
	mcpserver "github.com/FuradWho/Mix-MCP/internal/pkg/mcp-server"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	"log"
	"os"
)

func main() {
	fmt.Println("Hello World11")
	conf, err := config.ReadConfig("/Users/xin/Projects/Mix-MCP/static/config.json")
	if err != nil {
		log.Fatalln(err.Error())
	}
	logFile, err := os.OpenFile("/Users/xin/Projects/Mix-MCP/log.log", os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		log.Fatalln(err.Error())
	}
	log.SetOutput(logFile)
	mcpclient.InitAllClients(conf.McpServers)
	server, err := mcpclient.RegisterForServer()
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("grid_strategy", "在设定区间内划等距价格网格，价格下跌逐级买入、上涨逐级卖出，通过来回摆动赚取区间波动", mcpserver.GridHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("cancel_strategy", "cancel strategy", mcpserver.CancelStrategy)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("dca_strategy", "按固定时间间隔用同等法币金额市价买入，摊平成本、避免择时。", mcpserver.DCAHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("Passive_Maker", "在实时中间价的上下固定价差同时挂买单和卖单，靠“吃差价”获利", mcpserver.MakerHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("get_history_candles", "获取指定币对历史K线数据", mcpserver.HistoryCandleHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("get_symbol_price", "获取指定币对当前价格", mcpserver.CurrentPriceHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.Serve()
	if err != nil {
		log.Fatalln(err)
	}
	select {}
}
