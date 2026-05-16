// tts-elevenlabs synthesizes a sentence to an mp3 file using the
// ElevenLabs text-to-speech API.
//
// Requires:
//
//	ELEVENLABS_API_KEY  — your ElevenLabs API key
//
// Optional:
//
//	OUT_PATH  — destination mp3 path (default: out.mp3)
//	VOICE_ID  — ElevenLabs voice id (default: Rachel)
//
// Usage:
//
//	go run ./examples/tts-elevenlabs
//
// Output: writes the synthesized audio to OUT_PATH and prints the
// number of bytes written plus the audio Content-Type.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
)

const sampleText = "Hello from llmrouter. This audio was generated with the ElevenLabs API."

func main() {
	apiKey := os.Getenv("ELEVENLABS_API_KEY")
	if apiKey == "" {
		log.Fatal("ELEVENLABS_API_KEY required")
	}
	outPath := envOr("OUT_PATH", "out.mp3")
	voice := os.Getenv("VOICE_ID") // empty -> provider default (Rachel)

	provider, err := elevenlabs.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	ctx := context.Background()
	stream, err := provider.Speak(ctx, llmrouter.SpeechRequest{
		Input:  sampleText,
		Voice:  voice,
		Format: "mp3",
		Stream: true,
	})
	if err != nil {
		log.Fatalf("speak: %v", err)
	}

	out, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("create %s: %v", outPath, err)
	}
	defer out.Close()

	var written int
	for chunk := range stream.Chunks() {
		n, err := out.Write(chunk.Data)
		if err != nil {
			log.Fatalf("write: %v", err)
		}
		written += n
	}
	if err := stream.Err(); err != nil {
		log.Fatalf("stream: %v", err)
	}
	fmt.Printf("wrote %d bytes to %s (content-type=%s)\n", written, outPath, stream.ContentType)
}

// envOr returns the env value for name when non-empty; otherwise the fallback.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
