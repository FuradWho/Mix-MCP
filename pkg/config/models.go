package config

type McpServerConfig struct {
	Type    string   `json:"type"`
	Command string   `json:"command"`
	Args    []string `json:"args"`
	Env     []string `json:"env"`
}

type Config struct {
	McpServers []McpServerConfig `json:"mcp-servers"`
}
