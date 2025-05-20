package config

type McpServerConfig struct {
	Type    string   `json:"type"`
	Command string   `json:"command"`
	Args    []string `json:"args"`
	Env     []string `json:"env"`
	BaseUrl string   `json:"base_url"`
	Path    string   `json:"path"`
}

type Config struct {
	McpServers []McpServerConfig            `json:"mcp-servers"`
	Exchanges  map[string]map[string]string `json:"exchanges"`
}
