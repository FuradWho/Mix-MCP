package mcpclient

import (
	"context"
	"fmt"
	mcp "github.com/metoro-io/mcp-golang"
	"github.com/metoro-io/mcp-golang/transport/stdio"
	"io"
	"os/exec"
)

var clientsMap map[string]Tool

type McpServer interface {
	Register() error
}

type HttpServer struct {
}

type StdioServer struct {
	CommandName string
	CommandArgs []string
	Env         []string
	Stdin       *io.WriteCloser
	Stdout      *io.ReadCloser
}

func (server *StdioServer) Register() error {
	cmd := exec.Command(server.CommandName, server.CommandArgs...)
	cmd.Env = append(cmd.Env, server.Env...)
	serverStdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to get serverStdin pipe: %s", err.Error())
	}
	serverStdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get serverStdin pipe: %s", err.Error())
	}
	server.Stdin = &serverStdin
	server.Stdout = &serverStdout
	if err = cmd.Start(); err != nil {
		return fmt.Errorf("failed to start server with environment variables: %s", err.Error())
	}
	transport := stdio.NewStdioServerTransportWithIO(serverStdout, serverStdin)
	client := mcp.NewClient(transport)
	_, err = client.Initialize(context.Background())
	if err != nil {
		return fmt.Errorf("failed to initializet server: %s", err.Error())
	}
	cursor := ""
	tools, err := client.ListTools(context.Background(), &cursor)
	if err != nil {
		return err
	}
	for _, t := range tools.Tools {
		clientsMap[t.Name] = Tool{
			BelongClient: &Client{
				TransportType:  1,
				Server:         server,
				ClientInstance: client,
			},
			Name:        t.Name,
			Description: *t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return nil
}

func (server *HttpServer) Register() error {
	return nil
}

type Client struct {
	// 1:stdio 2:http
	TransportType  int
	Server         McpServer
	ClientInstance *mcp.Client
}

type Tool struct {
	BelongClient *Client
	Name         string
	Description  string
	InputSchema  interface{}
}

type ClientConfig struct {
}

func initAllClients(clients []ClientConfig) {
	//for _, v := range clients {
	//	v.Init()
	//}
}
