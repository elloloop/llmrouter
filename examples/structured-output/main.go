// structured-output asks OpenAI to extract a typed Person record from a
// short paragraph. The model is constrained by a JSON Schema supplied
// via ChatRequest.ResponseSchema; the assistant content is collected
// across the stream and unmarshaled into a Go struct.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key
//
// Usage:
//
//	go run ./examples/structured-output
//
// Output: the assistant's JSON, followed by the populated Go struct.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// Person is the typed shape we want the model to produce. The JSON
// schema below mirrors this struct; OpenAI guarantees the assistant
// output conforms when Strict is true.
type Person struct {
	Name      string   `json:"name"`
	AgeYears  int      `json:"age_years"`
	Hobbies   []string `json:"hobbies"`
	IsStudent bool     `json:"is_student"`
}

const personSchema = `{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "name":       {"type": "string"},
    "age_years":  {"type": "integer"},
    "hobbies":    {"type": "array", "items": {"type": "string"}},
    "is_student": {"type": "boolean"}
  },
  "required": ["name", "age_years", "hobbies", "is_student"]
}`

const passage = `Maria is twenty-seven years old. She studies marine biology and ` +
	`spends her weekends rock climbing and learning the piano.`

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}

	provider, err := openai.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	ctx := context.Background()
	stream, err := provider.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "Extract a Person record from the user passage. Reply with JSON only."),
			llmrouter.TextMessage("user", passage),
		},
		ResponseSchema: &llmrouter.ResponseSchema{
			Name:   "Person",
			Strict: true,
			Schema: json.RawMessage(personSchema),
		},
	})
	if err != nil {
		log.Fatalf("open stream: %v", err)
	}

	var raw strings.Builder
	for chunk := range stream.Chunks() {
		for _, choice := range chunk.Choices {
			raw.WriteString(choice.Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatalf("stream: %v", err)
	}

	fmt.Println("raw assistant JSON:")
	fmt.Println(raw.String())

	var person Person
	if err := json.Unmarshal([]byte(raw.String()), &person); err != nil {
		log.Fatalf("decode: %v", err)
	}
	fmt.Printf("\nparsed: %+v\n", person)
}
