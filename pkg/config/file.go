package config

import (
	"encoding/json"
	"os"
)

func ReadConfig(path string) (Config, error) {
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
