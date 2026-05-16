// stt-openai-whisper transcribes a local audio file using OpenAI
// Whisper (model whisper-1) and prints the recovered text plus any
// per-segment timings the upstream returned.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key
//	AUDIO_IN        — path to a local audio file (mp3, wav, m4a, webm,
//	                  flac, ogg, mp4). Defaults to "sample.mp3".
//
// Usage:
//
//	AUDIO_IN=./hello.mp3 go run ./examples/stt-openai-whisper
//
// Output: each segment (start..end seconds): "text", then a final
// "[final]" line carrying the concatenated transcript.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}
	audioPath := envOr("AUDIO_IN", "sample.mp3")

	f, err := os.Open(audioPath)
	if err != nil {
		log.Fatalf("open %s: %v", audioPath, err)
	}
	defer f.Close()

	provider, err := openai.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	ctx := context.Background()
	stream, err := provider.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:          "whisper-1",
		Audio:          f,
		AudioFormat:    mimeFromExt(audioPath),
		Filename:       filepath.Base(audioPath),
		ResponseFormat: "verbose_json",
	})
	if err != nil {
		log.Fatalf("transcribe: %v", err)
	}

	var full strings.Builder
	for seg := range stream.Segments() {
		if seg.Text == "" {
			continue
		}
		fmt.Printf("[%6.2fs..%6.2fs] %s\n",
			seg.Start.Seconds(), seg.End.Seconds(), seg.Text)
		if seg.Final {
			full.WriteString(seg.Text)
		} else {
			full.WriteString(seg.Text)
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatalf("stream: %v", err)
	}
	fmt.Println("[final]", strings.TrimSpace(full.String()))
}

// mimeFromExt guesses an audio MIME type from the file extension. Falls
// back to audio/mpeg so OpenAI still accepts the upload.
func mimeFromExt(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".mp3":
		return "audio/mpeg"
	case ".wav":
		return "audio/wav"
	case ".m4a", ".mp4":
		return "audio/mp4"
	case ".webm":
		return "audio/webm"
	case ".flac":
		return "audio/flac"
	case ".ogg":
		return "audio/ogg"
	default:
		return "audio/mpeg"
	}
}

// envOr returns the env value for name when non-empty; otherwise the fallback.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
