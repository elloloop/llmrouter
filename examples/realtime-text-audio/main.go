// realtime-text-audio opens an OpenAI Realtime WebSocket session, sends
// a text prompt, then drains the bidirectional event stream: response
// text deltas print to stdout, response audio deltas (PCM16) are
// concatenated and written to OUT_PATH for later playback.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key with Realtime access
//
// Optional:
//
//	OUT_PATH  — destination raw PCM16 file (default: out.pcm)
//	PROMPT    — what to ask the model (default: short stock prompt)
//
// Usage:
//
//	go run ./examples/realtime-text-audio
//
// The PCM16 output can be played back with ffplay:
//
//	ffplay -f s16le -ar 24000 -ac 1 out.pcm
//
// Output: streamed text on stdout, summary line on stderr noting the
// number of audio bytes written.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openairealtime"
)

const defaultPrompt = "Greet me in one short sentence and tell me what time of day suits a Go programmer best."

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}
	outPath := envOr("OUT_PATH", "out.pcm")
	prompt := envOr("PROMPT", defaultPrompt)

	provider, err := openairealtime.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	ctx := context.Background()
	session, err := provider.Connect(ctx, openairealtime.SessionConfig{
		Voice:             "alloy",
		Instructions:      "You are concise and friendly.",
		Modalities:        []string{"text", "audio"},
		OutputAudioFormat: "pcm16",
	})
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer session.Close()

	if err := session.SendText(ctx, prompt); err != nil {
		log.Fatalf("send text: %v", err)
	}

	out, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("create %s: %v", outPath, err)
	}
	defer out.Close()

	var audioBytes int
	for ev := range session.Events() {
		switch ev.Type {
		case "response.text.delta":
			fmt.Print(ev.Text)
		case "response.audio.delta":
			n, werr := out.Write(ev.AudioDelta)
			if werr != nil {
				log.Fatalf("write audio: %v", werr)
			}
			audioBytes += n
		case "response.done":
			// One turn is complete; tear down.
			_ = session.Close()
		case "error":
			log.Fatalf("upstream error: %v", ev.Error)
		}
	}
	if err := session.Err(); err != nil {
		log.Fatalf("session: %v", err)
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "wrote %d pcm16 bytes to %s\n", audioBytes, outPath)
}

// envOr returns the env value for name when non-empty; otherwise the fallback.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
