// gateway-passthrough is a minimal HTTP server that mirrors OpenAI's
// /v1/chat/completions endpoint and proxies the request to upstream
// OpenAI byte-identically. Inbound JSON is passed straight through via
// ChatRequest.Raw, and the chunk.Raw bytes from the stream are written
// back as SSE frames without re-marshaling. This is the building block
// for an LLM gateway / observability proxy that must not break the
// upstream wire format.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key (used by the gateway when
//	                  talking upstream)
//
// Optional:
//
//	ADDR  — listen address (default: :8080)
//
// Usage:
//
//	go run ./examples/gateway-passthrough
//
// Then, in another terminal:
//
//	curl -N http://localhost:8080/v1/chat/completions \
//	  -H 'content-type: application/json' \
//	  -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}'
//
// You will see OpenAI's SSE frames (data: {...}\n\n ... data: [DONE])
// passing through byte-for-byte.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}
	addr := envOr("ADDR", ":8080")

	provider, err := openai.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	http.HandleFunc("/v1/chat/completions", handler(provider))
	log.Printf("gateway listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}

// handler returns an HTTP handler that forwards the inbound JSON body
// to OpenAI via CompletionStream, then writes each chunk.Raw back as an
// SSE frame. Flushes after every frame so curl --no-buffer prints the
// stream in real time.
func handler(provider *openai.Provider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Pick out the model so the typed request is well-formed; the
		// rest of the fields ride along in Raw and are forwarded
		// unchanged by the openai provider.
		var probe struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &probe); err != nil {
			http.Error(w, "decode body: "+err.Error(), http.StatusBadRequest)
			return
		}

		stream, err := provider.CompletionStream(r.Context(), llmrouter.ChatRequest{
			Model: probe.Model,
			Raw:   body,
		})
		if err != nil {
			http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, _ := w.(http.Flusher)

		for chunk := range stream.Chunks() {
			payload := chunk.Raw
			if len(payload) == 0 {
				// Fall back to typed JSON for providers that don't
				// populate Raw (not the case for openai, but defensive).
				payload, _ = json.Marshal(chunk)
			}
			fmt.Fprintf(w, "data: %s\n\n", payload)
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err := stream.Err(); err != nil {
			log.Printf("stream error: %v", err)
			return
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// envOr returns the env value for name when non-empty; otherwise the fallback.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
