// voice-agent-loop wires three providers together into a single
// turn-taking voice agent: Deepgram for STT, OpenAI for chat, and
// ElevenLabs streaming TTS for the spoken reply.
//
// PLACEHOLDER: this example does NOT capture from a microphone. The
// "user audio" is read from AUDIO_IN (a local file) and Deepgram's
// streaming endpoint sends it as if it were live mic input. The
// generated TTS audio is written to AUDIO_OUT. In a real agent you
// would replace the file reads/writes with PortAudio or similar.
//
// Requires:
//
//	DEEPGRAM_API_KEY    — for Deepgram STT
//	OPENAI_API_KEY      — for the chat reply
//	ELEVENLABS_API_KEY  — for the spoken reply
//	AUDIO_IN            — path to a local wav/mp3 file standing in for
//	                      mic input
//
// Optional:
//
//	AUDIO_OUT  — destination mp3 for the synthesised reply (default: reply.mp3)
//	MODEL      — OpenAI chat model (default: gpt-4o-mini)
//
// Usage:
//
//	AUDIO_IN=./hello.mp3 go run ./examples/voice-agent-loop
//
// Output: logs the final user transcript, streams the assistant reply
// to stdout, and writes the TTS audio to AUDIO_OUT.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/deepgram"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
	"github.com/elloloop/llmrouter/providers/openai"
)

func main() {
	deepgramKey := mustEnv("DEEPGRAM_API_KEY")
	openaiKey := mustEnv("OPENAI_API_KEY")
	elevenKey := mustEnv("ELEVENLABS_API_KEY")
	audioIn := mustEnv("AUDIO_IN")
	audioOut := envOr("AUDIO_OUT", "reply.mp3")
	chatModel := envOr("MODEL", "gpt-4o-mini")

	ctx := context.Background()

	// 1) Deepgram: streaming transcription of the input audio file.
	userText, err := transcribe(ctx, deepgramKey, audioIn)
	if err != nil {
		log.Fatalf("transcribe: %v", err)
	}
	if userText == "" {
		log.Fatal("empty transcript from deepgram")
	}
	log.Printf("user said: %q", userText)

	// 2) OpenAI: chat reply, streamed token-by-token. We also accumulate
	//    the full reply for the TTS step.
	openaiProvider, err := openai.New(llmrouter.WithAPIKey(openaiKey))
	if err != nil {
		log.Fatalf("openai: %v", err)
	}
	chatStream, err := openaiProvider.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: chatModel,
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "You are a concise voice agent. Keep replies under 30 words."),
			llmrouter.TextMessage("user", userText),
		},
	})
	if err != nil {
		log.Fatalf("chat stream: %v", err)
	}

	// 3) ElevenLabs realtime: open the TTS context up front so we can
	//    pipe chat tokens into it as they arrive. The audio side is
	//    drained on a goroutine into AUDIO_OUT concurrently.
	elevenProvider, err := elevenlabs.New(llmrouter.WithAPIKey(elevenKey))
	if err != nil {
		log.Fatalf("elevenlabs: %v", err)
	}
	audioStream, rtCtx, err := elevenProvider.SpeakRealtime(ctx, llmrouter.SpeechRequest{
		Format: "mp3",
		Stream: true,
	})
	if err != nil {
		log.Fatalf("eleven realtime: %v", err)
	}

	audioDone := make(chan struct{})
	var audioBytes int
	go func() {
		defer close(audioDone)
		out, err := os.Create(audioOut)
		if err != nil {
			log.Printf("create %s: %v", audioOut, err)
			return
		}
		defer out.Close()
		for ch := range audioStream.Chunks() {
			n, werr := out.Write(ch.Data)
			if werr != nil {
				log.Printf("write audio: %v", werr)
				return
			}
			audioBytes += n
		}
		if err := audioStream.Err(); err != nil {
			log.Printf("audio stream: %v", err)
		}
	}()

	var fullReply strings.Builder
	for chunk := range chatStream.Chunks() {
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == "" {
				continue
			}
			fmt.Print(choice.Delta.Content)
			fullReply.WriteString(choice.Delta.Content)
			if err := rtCtx.Append(ctx, choice.Delta.Content); err != nil {
				log.Printf("tts append: %v", err)
			}
		}
	}
	if err := chatStream.Err(); err != nil {
		log.Fatalf("chat stream: %v", err)
	}
	fmt.Println()

	if err := rtCtx.Finalize(ctx); err != nil {
		log.Printf("finalize tts: %v", err)
	}
	<-audioDone
	log.Printf("wrote %d audio bytes to %s", audioBytes, audioOut)
}

// transcribe sends the file at path through Deepgram's streaming
// endpoint and returns the concatenated final transcript text. We rely
// on Final segments so interim hypotheses don't get double-counted.
func transcribe(ctx context.Context, apiKey, path string) (string, error) {
	provider, err := deepgram.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		return "", err
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	stream, err := provider.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:       "nova-3",
		Audio:       f,
		AudioFormat: "audio/mpeg",
		Stream:      true,
	})
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for seg := range stream.Segments() {
		if seg.Final && seg.Text != "" {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(seg.Text)
		}
	}
	if err := stream.Err(); err != nil {
		return "", err
	}
	return strings.TrimSpace(b.String()), nil
}

// mustEnv reads a required env var or aborts with a clear message.
func mustEnv(name string) string {
	v := os.Getenv(name)
	if v == "" {
		log.Fatalf("%s required", name)
	}
	return v
}

// envOr returns the env value for name when non-empty; otherwise the fallback.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
