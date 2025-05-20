package mcpclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/FuradWho/Mix-MCP/pkg/mcp-golang/transport/http"
	"io"
	"log"
	"os/exec"

	"github.com/FuradWho/Mix-MCP/pkg/config"
	mcp "github.com/FuradWho/Mix-MCP/pkg/mcp-golang"
	"github.com/FuradWho/Mix-MCP/pkg/mcp-golang/transport/stdio"
	"github.com/invopop/jsonschema"
)

var clientsMap map[string]Tool

type McpServer interface {
	Register() error
}

type HttpServer struct {
	BaseUrl string
	Path    string
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
	transport := http.NewHTTPClientTransport(server.Path).WithBaseURL(server.BaseUrl)
	client := mcp.NewClient(transport)
	_, err := client.Initialize(context.Background())

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
		} else if s.Type == "http" {
			server := &HttpServer{
				BaseUrl: s.BaseUrl,
				Path:    s.Path,
			}
			err = server.Register()
			if err != nil {
				log.Println(err.Error())
			}
		}
	}
}

func toolTextResponse(text string) *mcp.ToolResponse {
	return mcp.NewToolResponse(mcp.NewTextContent(text))
}

func SchemaFromInterface(schema interface{}) (*jsonschema.Schema, error) {
	switch schema := schema.(type) {
	case []byte:
		return BytesToJSONSchema(schema)
	case *jsonschema.Schema:
		return schema, nil
	case map[string]interface{}:
		bytes, err := json.Marshal(schema)
		if err != nil {
			return nil, err
		}
		return BytesToJSONSchema(bytes)
	default:
		log.Println("Unsupported schema type")
		return nil, errors.New("unsupported schema type")
	}
}

func RegisterForServer() (*mcp.Server, error) {
	var err error
	server := mcp.NewServer(stdio.NewStdioServerTransport())
	for name, v := range clientsMap {
		handler := func(arguments any) (*mcp.ToolResponse, error) {
			return v.BelongClient.ClientInstance.CallTool(context.Background(), name, arguments)
		}

		schema, err := SchemaFromInterface(v.InputSchema)
		if err != nil {
			return nil, err
		}

		err = server.RegisterToolWithInputSchema(name, v.Description, handler, schema)
		if err != nil {
			return nil, err
		}
	}
	return server, err
}
