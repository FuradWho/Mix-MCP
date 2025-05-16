package main

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"strings"
	"unicode"
)

const rawSchema = `{
  "$schema":"http://json-schema.org/draft-07/schema#",
  "additionalProperties":false,
  "properties":{
    "address":{
      "description":"The Ethereum address to query",
      "type":"string"
    },
    "chainId":{
      "description":"Optional. The chain ID to use. If provided with a named network and they don't match, the RPC's chain ID will be used.",
      "type":"number"
    },
    "provider":{
      "description":"Optional. Either a network name or custom RPC URL. Use getAllNetworks to see available networks and their details, or getNetwork to get info about a specific network. You can use any network name returned by these tools as a provider value.",
      "type":"string"
    }
  },
  "required":["address"],
  "type":"object"
}`

// ---------------------------------------------------------------------
// 解析 JSON-Schema（只关心 properties / required）
type prop struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}
type schema struct {
	Properties map[string]prop `json:"properties"`
	Required   []string        `json:"required"`
}

// ---------------------------------------------------------------------
// 主流程
func main() {
	// 1. 反序列化 Schema
	var sc schema
	if err := json.Unmarshal([]byte(rawSchema), &sc); err != nil {
		log.Fatalf("decode schema: %v", err)
	}

	// 2. 构造结构体类型
	t := buildStructType(sc)

	// 3. 创建实例并随便赋点值
	v := reflect.New(t).Elem()
	v.FieldByName("Address").SetString("0xAbC123...")
	v.FieldByName("ChainId").SetFloat(1) // float64
	// Provider 留空，omitempty

	// 4. 打印序列化结果
	out, _ := json.MarshalIndent(v.Interface(), "", "  ")
	fmt.Println("序列化后的 JSON：")
	fmt.Println(string(out))

	// 5. 打印字段及 tag，验证格式
	fmt.Println("\n字段与 tag：")
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		fmt.Printf("%-8s %s\n", f.Name+":", f.Tag)
	}
}

// ---------------------------------------------------------------------
// 根据 schema 生成 reflect.Type（匿名结构体）
func buildStructType(sc schema) reflect.Type {
	required := make(map[string]bool, len(sc.Required))
	for _, r := range sc.Required {
		required[r] = true
	}

	var fields []reflect.StructField
	for jsonKey, p := range sc.Properties {
		fields = append(fields, reflect.StructField{
			Name: export(jsonKey), // 导出名，CamelCase
			Type: js2go(p.Type),   // Go 类型
			Tag:  makeTag(jsonKey, p.Description, required[jsonKey]),
		})
	}
	return reflect.StructOf(fields)
}

// ---------------------------------------------------------------------
// tag 生成：json + jsonschema
func makeTag(jsonKey, desc string, isReq bool) reflect.StructTag {
	// json:"name[,omitempty]"
	jTag := jsonKey
	if !isReq {
		jTag += ",omitempty"
	}

	// jsonschema:"required,description=..." 或 "description=..."
	var sb strings.Builder
	if isReq {
		sb.WriteString("required,")
	}
	sb.WriteString("description=")
	sb.WriteString(escapeQuotes(desc))

	tag := fmt.Sprintf(`json:"%s" jsonschema:"%s"`, jTag, sb.String())
	return reflect.StructTag(tag)
}

func escapeQuotes(s string) string {
	return strings.ReplaceAll(s, `"`, "'")
}

// ---------------------------------------------------------------------
// 工具函数
func js2go(t string) reflect.Type {
	switch t {
	case "string":
		return reflect.TypeOf("")
	case "number":
		return reflect.TypeOf(float64(0))
	case "integer":
		return reflect.TypeOf(int64(0))
	case "boolean":
		return reflect.TypeOf(true)
	default:
		return reflect.TypeOf(new(interface{})).Elem()
	}
}

func export(s string) string { // snake/kebab-case → CamelCase
	var out []rune
	up := true
	for _, r := range s {
		if r == '_' || r == '-' {
			up = true
			continue
		}
		if up {
			out = append(out, unicode.ToUpper(r))
			up = false
		} else {
			out = append(out, r)
		}
	}
	return string(out)
}
