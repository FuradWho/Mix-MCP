package strategy

import (
	"context"
	"fmt"
	"strings"
)

var StrategyMap = make(map[string]context.CancelFunc)

func PushStrategyCtx(name string, cancelFunc context.CancelFunc) error {
	if _, ok := StrategyMap[name]; ok {
		return fmt.Errorf("%s already exists", name)
	}
	StrategyMap[name] = cancelFunc
	return nil
}

func PopStrategyCtx(name string) {

	if cancelFunc, ok := StrategyMap[name]; ok {
		cancelFunc()
	}
	delete(StrategyMap, name)
}

func splitString(r rune) bool {
	return r == '_' || r == '-'
}

// SplitStringChar 字符串按照多个字符分割
func SplitStringChar(s string) []string {
	a := strings.FieldsFunc(s, splitString)
	return a
}

func FormatSymbol(exchange, symbol string) string {
	symbols := SplitStringChar(symbol)
	if len(symbols) == 0 {
		return ""
	}
	if len(symbols) == 1 && strings.Contains(symbol, "USDT") {
		tokens := strings.Split(symbol, "USDT")
		symbols[0] = tokens[0]
		symbols = append(symbols, "USDT")
	}

	newSymbol := func(s string) string {
		return symbols[0] + s + symbols[1]
	}

	switch exchange {
	case "bitget":
		return newSymbol("")
	default:

		return symbol
	}
}
