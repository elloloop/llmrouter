//go:build integration

// Package vertex integration tests exercise a real Vertex AI endpoint and
// are excluded from the default `go test` run. Enable with:
//
//	GOOGLE_CLOUD_PROJECT=my-proj \
//	GOOGLE_CLOUD_LOCATION=us-central1 \
//	go test -tags=integration ./providers/vertex/...
//
// The test requires Application Default Credentials (ADC) — run
// `gcloud auth application-default login` first, or set
// GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file.
package vertex

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

func TestIntegration_CompletionStream(t *testing.T) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	region := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if project == "" || region == "" {
		t.Skip("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION required for integration tests")
	}

	p, err := New(WithProject(project), WithRegion(region))
	if err != nil {
		t.Fatalf("provider New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", "Say the word 'pong' and nothing else."),
		},
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	var buf strings.Builder
	var sawFinish bool
	for chunk := range stream.Chunks() {
		if len(chunk.Choices) == 0 {
			continue
		}
		buf.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != "" {
			sawFinish = true
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream terminated with error: %v", err)
	}
	if !sawFinish {
		t.Fatalf("never observed a finish_reason chunk")
	}
	if buf.Len() == 0 {
		t.Fatalf("no content streamed")
	}
}
