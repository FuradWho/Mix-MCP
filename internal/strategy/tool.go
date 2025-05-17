package strategy

import (
	"context"
	"fmt"
)

var StrategyMap = make(map[string]context.Context)

func PushStrategyCtx(name string, ctx context.Context) error {
	if _, ok := StrategyMap[name]; ok {
		return fmt.Errorf("%s already exists", name)
	}
	StrategyMap[name] = ctx
	return nil
}

func PopStrategyCtx(name string) {
	delete(StrategyMap, name)
}
