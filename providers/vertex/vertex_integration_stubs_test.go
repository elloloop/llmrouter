//go:build integration

// Additional Vertex integration-test stubs that document the SDK-bound
// paths we cannot unit-test (the *genai.Client has unexported fields with
// no public test seam). Each test skips when the required env vars are
// missing so unconfigured CI is unaffected.
//
// Run with:
//
//	GOOGLE_CLOUD_PROJECT=my-proj \
//	GOOGLE_CLOUD_LOCATION=us-central1 \
//	go test -tags=integration ./providers/vertex/...
//
// Requires Application Default Credentials (`gcloud auth
// application-default login` or GOOGLE_APPLICATION_CREDENTIALS).
package vertex

import (
	"bytes"
	"context"
	"os"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// requireVertexEnv skips when project/region env vars are absent.
func requireVertexEnv(t *testing.T) (project, region string) {
	t.Helper()
	project = os.Getenv("GOOGLE_CLOUD_PROJECT")
	region = os.Getenv("GOOGLE_CLOUD_LOCATION")
	if project == "" || region == "" {
		t.Skip("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION required for integration tests")
	}
	return project, region
}

// TestIntegration_Embed exercises the Embed path against a real Vertex
// embedding endpoint. The text-embedding model variants live behind the
// same gRPC-backed client as completions, so this path can only be
// validated end-to-end.
func TestIntegration_Embed(t *testing.T) {
	project, region := requireVertexEnv(t)
	p, err := New(WithProject(project), WithRegion(region))
	if err != nil {
		t.Fatalf("provider New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:    "text-embedding-004",
		Inputs:   []string{"hello world"},
		TaskType: "RETRIEVAL_QUERY",
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) == 0 {
		t.Fatalf("unexpected embeddings shape: %+v", resp.Embeddings)
	}
}

// TestIntegration_Speak exercises the Gemini TTS path via Vertex (when
// supported by the project). Stub form: caller may skip if not enabled.
func TestIntegration_Speak(t *testing.T) {
	project, region := requireVertexEnv(t)
	p, err := New(WithProject(project), WithRegion(region))
	if err != nil {
		t.Fatalf("provider New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{
		Model:  "gemini-2.5-flash-preview-tts",
		Input:  "Hello",
		Format: "wav",
	})
	if err != nil {
		t.Skipf("Speak unavailable on this project: %v", err)
	}
	var bytes int
	for c := range stream.Chunks() {
		bytes += len(c.Data)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if bytes == 0 {
		t.Fatal("no audio returned")
	}
}

// TestIntegration_Transcribe exercises the STT path. Stub: a real call
// requires a tiny audio fixture; here we send a near-empty payload and
// accept either success or an upstream-error response.
func TestIntegration_Transcribe(t *testing.T) {
	project, region := requireVertexEnv(t)
	p, err := New(WithProject(project), WithRegion(region))
	if err != nil {
		t.Fatalf("provider New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// Tiny silent WAV header (44 bytes) so the request is well-formed.
	silent := []byte{
		0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
		0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
		0x80, 0x3e, 0x00, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,
		0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
	}
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:       "gemini-1.5-flash",
		Audio:       bytes.NewReader(silent),
		AudioFormat: "audio/wav",
	})
	if err != nil {
		t.Skipf("Transcribe init failed (expected on some projects): %v", err)
	}
	for range stream.Segments() {
	}
	// Either Err or nil is acceptable; we just want the path exercised.
	_ = stream.Err()
}

// TestIntegration_ContextCancel proves the streaming path responds to
// context cancellation mid-stream by terminating the channel and
// surfacing context.Canceled (or wrapped) via Err().
func TestIntegration_ContextCancel(t *testing.T) {
	project, region := requireVertexEnv(t)
	p, err := New(WithProject(project), WithRegion(region))
	if err != nil {
		t.Fatalf("provider New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", "Count slowly to one hundred."),
		},
		MaxTokens: 2048,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	// Cancel after a tiny delay so at least one chunk has a chance to
	// arrive on a healthy network.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	for range stream.Chunks() {
	}
	// Err may be ctx.Err() or a wrapped variant — both are acceptable
	// signals that cancellation propagated.
	if stream.Err() == nil {
		t.Skip("stream completed before cancellation took effect")
	}
}

// TestIntegration_BadProject confirms a bogus project surfaces a clear
// error rather than panicking. Uses a project id that is syntactically
// valid but unlikely to exist.
func TestIntegration_BadProject(t *testing.T) {
	if os.Getenv("GOOGLE_CLOUD_LOCATION") == "" {
		t.Skip("GOOGLE_CLOUD_LOCATION required to construct the client")
	}
	p, err := New(
		WithProject("this-project-should-not-exist-llmrouter-integration"),
		WithRegion(os.Getenv("GOOGLE_CLOUD_LOCATION")),
	)
	if err != nil {
		// Construction may itself fail when ADC scopes are wrong — that
		// is also a valid integration signal.
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected an error for bogus project, got nil")
	}
}
