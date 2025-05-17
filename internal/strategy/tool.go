package strategy

import (
	"context"
	"fmt"
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
