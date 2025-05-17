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
	conf, err := config.ReadConfig("/Users/furad/Projects/Mix-MCP/static/config.json")
	if err != nil {
		log.Fatalln(err.Error())
	}
	logFile, err := os.OpenFile("/Users/xin/Projects/Mix-MCP/log.log", os.O_CREATE|os.O_APPEND, 0777)
	if err != nil {
		log.Fatalln(err.Error())
	}
	log.SetOutput(logFile)
	mcpclient.InitAllClients(conf.McpServers)
	server, err := mcpclient.RegisterForServer()
	if err != nil {
		log.Fatalln(err)
	}
	err = server.RegisterTool("grid_strategy", "grid strategy", mcpserver.GridHandler)
	if err != nil {
		log.Fatalln(err)
	}
	err = server.Serve()
	if err != nil {
		log.Fatalln(err)
	}
	select {}
}
