package schema

import (
	"fmt"
	"strings"
	"unicode"
)

// -------- 数据结构 --------
type Field struct {
	Description string `json:"description"`
	Type        string `json:"type,omitempty"`
	Pattern     string `json:"pattern,omitempty"`
	Items       *Field `json:"items,omitempty"`
}

type Schema = map[string]Field

// -------- 解析入口 --------
func ParseSchema(raw string) (Schema, error) {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "map[") {
		return nil, fmt.Errorf("input must start with 'map['")
	}
	idx := 0
	node, err := parseMap(raw, &idx)
	if err != nil {
		return nil, err
	}
	// 将通用 map[string]any 转为 Schema
	return toSchema(node), nil
}

// -------- 递归解析 --------
func parseMap(s string, i *int) (map[string]any, error) {
	expect(s, i, "map[")
	out := make(map[string]any)
	for {
		skipSpaces(s, i)
		if peek(s, *i) == ']' { // 结束
			*i++
			return out, nil
		}

		key := readToken(s, i)
		expectRune(s, i, ':')
		val, err := parseValue(s, i)
		if err != nil {
			return nil, err
		}
		out[key] = val
	}
}

func parseValue(s string, i *int) (any, error) {
	if strings.HasPrefix(s[*i:], "map[") {
		return parseMap(s, i)
	}
	// 原子字符串，一直读到空格或 ']'
	return readToken(s, i), nil
}

// --------— util --------
func readToken(s string, i *int) string {
	start := *i
	for *i < len(s) && !unicode.IsSpace(rune(s[*i])) && s[*i] != ']' {
		*i++
	}
	return s[start:*i]
}

func expect(s string, i *int, lit string) {
	if !strings.HasPrefix(s[*i:], lit) {
		panic("syntax error")
	}
	*i += len(lit)
}

func expectRune(s string, i *int, r byte) {
	if peek(s, *i) != r {
		panic("syntax error")
	}
	*i++
}

func skipSpaces(s string, i *int) {
	for *i < len(s) && unicode.IsSpace(rune(s[*i])) {
		*i++
	}
}

func peek(s string, i int) byte { return s[i] }

// -------- map[string]any → Field 递归收敛 --------
func toSchema(m map[string]any) Schema {
	out := make(Schema, len(m))
	for k, v := range m {
		out[k] = toField(v.(map[string]any))
	}
	return out
}

func toField(m map[string]any) Field {
	f := Field{}
	for k, v := range m {
		switch k {
		case "description":
			f.Description = v.(string)
		case "type":
			f.Type = v.(string)
		case "pattern":
			f.Pattern = v.(string)
		case "items":
			f.Items = ptr(toField(v.(map[string]any)))
		default: // 继续向下递归，允许更深层级
			// 可按需扩展
		}
	}
	return f
}

func ptr[T any](v T) *T { return &v }
