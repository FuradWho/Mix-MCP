package mcpclient

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
	"github.com/FuradWho/Mix-MCP/pkg/mcp-golang/transport/stdio"
	"github.com/invopop/jsonschema"
	"io"
	"log"
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

func BytesToJSONSchema(data []byte) (*jsonschema.Schema, error) {
	schema := &jsonschema.Schema{}
	err := json.Unmarshal(data, schema)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON schema: %w", err)
	}
	return schema, nil
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
		var toolDescription string
		if t.Description == nil {
			toolDescription = t.Name
			//continue
		} else {
			toolDescription = *t.Description
		}
		clientsMap[t.Name] = Tool{
			BelongClient: &Client{
				TransportType:  1,
				Server:         server,
				ClientInstance: client,
			},
			Name:        t.Name,
			Description: toolDescription,
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

func InitAllClients(servers []config.McpServerConfig) {
	var err error
	clientsMap = make(map[string]Tool, len(servers))
	for _, s := range servers {
		if s.Type == "stdio" {
			server := &StdioServer{
				CommandName: s.Command,
				CommandArgs: s.Args,
				Env:         s.Env,
			}

			err = server.Register()
			if err != nil {
				log.Println(err.Error())
			}

			fmt.Println(s)
		}
	}
}

func toolTextResponse(text string) *mcp.ToolResponse {
	return mcp.NewToolResponse(mcp.NewTextContent(text))
}

func RegisterForServer() (*mcp.Server, error) {
	var err error
	server := mcp.NewServer(stdio.NewStdioServerTransport())
	for name, v := range clientsMap {
		handler := func(arguments any) (*mcp.ToolResponse, error) {
			return v.BelongClient.ClientInstance.CallTool(context.Background(), name, arguments)
		}

		//log.Printf("Tool %s schema: %+v", name, jsonschema.Reflect(v.InputSchema))
		//inputSchema := v.InputSchema.(jsonschema.Schema)
		//jsonschema.Reflect(inputSchema)

		var schema *jsonschema.Schema

		switch inputSchema := v.InputSchema.(type) {
		case []byte:
			// 直接从 bytes 反序列化
			schema, err = BytesToJSONSchema(inputSchema)
			if err != nil {
				log.Printf("Failed to unmarshal schema for tool %s: %v", name, err)
				continue
			}

		case *jsonschema.Schema:
			// 已经是 Schema 类型
			schema = inputSchema

		case map[string]interface{}:
			// 如果是 map，先转换成 JSON bytes
			bytes, err := json.Marshal(inputSchema)
			if err != nil {
				log.Printf("Failed to marshal schema map for tool %s: %v", name, err)
				continue
			}
			schema, err = BytesToJSONSchema(bytes)
			if err != nil {
				log.Printf("Failed to unmarshal schema bytes for tool %s: %v", name, err)
				continue
			}

		default:
			log.Printf("Unsupported schema type for tool %s: %T", name, v.InputSchema)
			continue
		}

		// 确保 Properties 不为 nil
		//if schema.Properties == nil {
		//	schema.Properties = make(map[string]*jsonschema.Schema)
		//}

		err = server.RegisterToolWithInputSchema(name, v.Description, handler, schema)
		if err != nil {
			log.Println(err)
		}
	}
	return server, err
}
