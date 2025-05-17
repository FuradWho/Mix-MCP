package main

import (
	"fmt"
	mcpclient "github.com/FuradWho/Mix-MCP/internal/pkg/mcp-client"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	"log"
)

func main() {
	fmt.Println("Hello World11")
	conf, err := config.ReadConfig("/Users/furad/Projects/Mix-MCP/static/config.json")
	if err != nil {
		log.Fatalln(err.Error())
	}
	mcpclient.InitAllClients(conf.McpServers)
	server, err := mcpclient.RegisterForServer()
	if err != nil {
		log.Fatalln(err)
	}
	err = server.Serve()
	if err != nil {
		log.Fatalln(err)
	}
	select {}
}
