package config

import (
	"encoding/json"
	"errors"
	"os"
)

var configPath = ""

func InitConfigPath(path string) {
	configPath = path
}

func ReadConfig(path string) (Config, error) {
	if path != configPath {
		configPath = path
	}
	file, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var c Config
	err = json.Unmarshal(file, &c)
	if err != nil {
		return Config{}, err
	}
	return c, nil
}

func ReadExchangeConfig(exchangeName string) (map[string]string, error) {
	file, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	var c Config
	err = json.Unmarshal(file, &c)
	if err != nil {
		return nil, err
	}
	ex, ok := c.Exchanges[exchangeName]
	if !ok {
		return nil, errors.New("exchange not found")
	}
	return ex, nil
}
