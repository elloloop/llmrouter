package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// fakeAudioServer returns an httptest.Server that uses the given handler.
func fakeAudioServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

func newAudioProvider(t *testing.T, baseURL string) *openai.Provider {
	t.Helper()
	p, err := openai.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(baseURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// ---------- Speak (TTS) ----------

func speechBytes() []byte { return []byte("MP3-AUDIO-DATA") }

func speechHandler(t *testing.T, body []byte, inspect func(*testing.T, *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if inspect != nil {
			inspect(t, r)
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}
}

func TestSpeak_NonStreaming_SingleChunk(t *testing.T) {
	want := speechBytes()
	srv := fakeAudioServer(t, speechHandler(t, want, nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hello"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	var got []byte
	count := 0
	for chunk := range stream.Chunks() {
		got = append(got, chunk.Data...)
		count++
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if count != 1 {
		t.Errorf("chunk count = %d, want 1", count)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("audio = %q, want %q", got, want)
	}
}

func TestSpeak_ContentTypeFromHeader(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/opus")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("opus"))
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if stream.ContentType != "audio/opus" {
		t.Errorf("ContentType = %q, want audio/opus", stream.ContentType)
	}
	for range stream.Chunks() {
	}
	_ = stream.Err()
}

func TestSpeak_URLPath(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Path
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	if seen != "/audio/speech" {
		t.Errorf("path = %q, want /audio/speech", seen)
	}
}

func TestSpeak_Authorization(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q", seen)
	}
}

func TestSpeak_FormatDefaultsToMP3(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["response_format"] != "mp3" {
		t.Errorf("response_format = %v, want mp3", m["response_format"])
	}
}

func TestSpeak_FormatExplicit(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Format: "wav"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["response_format"] != "wav" {
		t.Errorf("response_format = %v, want wav", m["response_format"])
	}
}

func TestSpeak_SpeedPointer(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	speed := 1.5
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Speed: &speed})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["speed"].(float64) != 1.5 {
		t.Errorf("speed = %v, want 1.5", m["speed"])
	}
}

func TestSpeak_VoiceForwarded(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Voice: "alloy"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["voice"] != "alloy" {
		t.Errorf("voice = %v, want alloy", m["voice"])
	}
}

func TestSpeak_RawOverlay(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw := json.RawMessage(`{"vendor":"x","model":"old","input":"old"}`)
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "new-input", Raw: raw})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["vendor"] != "x" {
		t.Errorf("vendor extra dropped: %v", m)
	}
	if m["model"] != "tts-1" {
		t.Errorf("model overlay: %v", m["model"])
	}
	if m["input"] != "new-input" {
		t.Errorf("input overlay: %v", m["input"])
	}
}

func TestSpeak_Streaming_MultipleChunks(t *testing.T) {
	// Body large enough to require multiple 8KiB reads.
	big := bytes.Repeat([]byte("A"), 20*1024)
	srv := fakeAudioServer(t, speechHandler(t, big, nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Stream: true})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	var got []byte
	chunks := 0
	for chunk := range stream.Chunks() {
		got = append(got, chunk.Data...)
		chunks++
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if chunks < 2 {
		t.Errorf("chunks = %d, want >=2 for streamed body", chunks)
	}
	if !bytes.Equal(got, big) {
		t.Errorf("audio bytes mismatch len got=%d want=%d", len(got), len(big))
	}
}

func TestSpeak_4xx_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upstream.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", upstream.StatusCode)
	}
	if upstream.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", upstream.Provider)
	}
}

func TestSpeak_429_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"slow"}`))
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) || upstream.StatusCode != 429 {
		t.Fatalf("err = %v, want 429 ErrUpstream", err)
	}
}

func TestSpeak_ChunkRawEqualsData(t *testing.T) {
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for chunk := range stream.Chunks() {
		if !bytes.Equal(chunk.Data, chunk.Raw) {
			t.Errorf("Data != Raw: %q vs %q", chunk.Data, chunk.Raw)
		}
	}
}

func TestSpeak_BodyContainsModelAndInput(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1-hd", Input: "say"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["model"] != "tts-1-hd" {
		t.Errorf("model = %v", m["model"])
	}
	if m["input"] != "say" {
		t.Errorf("input = %v", m["input"])
	}
}

// ---------- Transcribe (STT) ----------

func transcribeVerboseJSON() string {
	return `{
		"text":"hello world",
		"segments":[{"text":"hello world","start":0.0,"end":1.2}],
		"words":[
			{"word":"hello","start":0.0,"end":0.6},
			{"word":"world","start":0.6,"end":1.2}
		]
	}`
}

func transcribeHandler(t *testing.T, status int, body string, inspect func(*testing.T, *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if inspect != nil {
			inspect(t, r)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func TestTranscribe_HappyPath_VerboseJSON(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("AUDIO")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var segments []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segments = append(segments, s)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if len(segments) != 1 {
		t.Fatalf("segments = %d, want 1", len(segments))
	}
	if !segments[0].Final {
		t.Errorf("Final = false, want true")
	}
	if segments[0].Text != "hello world" {
		t.Errorf("Text = %q", segments[0].Text)
	}
	if len(segments[0].Words) != 2 {
		t.Errorf("Words len = %d, want 2", len(segments[0].Words))
	}
	if segments[0].Words[0].Word != "hello" {
		t.Errorf("Words[0] = %v", segments[0].Words[0])
	}
	if segments[0].End != 1200*time.Millisecond {
		t.Errorf("End = %v, want 1.2s", segments[0].End)
	}
}

func TestTranscribe_URLPath(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Path
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("AUDIO")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if seen != "/audio/transcriptions" {
		t.Errorf("path = %q", seen)
	}
}

func TestTranscribe_MultipartFields(t *testing.T) {
	type parsed struct {
		model          string
		language       string
		prompt         string
		responseFormat string
		fileFieldFound bool
		fileFilename   string
		fileContent    []byte
	}
	var got parsed

	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		mr, err := r.MultipartReader()
		if err != nil {
			t.Errorf("MultipartReader: %v", err)
			return
		}
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Errorf("NextPart: %v", err)
				return
			}
			data, _ := io.ReadAll(part)
			switch part.FormName() {
			case "file":
				got.fileFieldFound = true
				got.fileFilename = part.FileName()
				got.fileContent = data
			case "model":
				got.model = string(data)
			case "language":
				got.language = string(data)
			case "prompt":
				got.prompt = string(data)
			case "response_format":
				got.responseFormat = string(data)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()

	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:    "whisper-1",
		Audio:    bytes.NewReader([]byte("AUDIO-PAYLOAD")),
		Language: "en",
		Prompt:   "context",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}

	if !got.fileFieldFound {
		t.Errorf("file field missing")
	}
	if string(got.fileContent) != "AUDIO-PAYLOAD" {
		t.Errorf("file content = %q", got.fileContent)
	}
	if got.model != "whisper-1" {
		t.Errorf("model = %q", got.model)
	}
	if got.language != "en" {
		t.Errorf("language = %q", got.language)
	}
	if got.prompt != "context" {
		t.Errorf("prompt = %q", got.prompt)
	}
	if got.responseFormat != "verbose_json" {
		t.Errorf("response_format default = %q, want verbose_json", got.responseFormat)
	}
}

func TestTranscribe_FilenameDefault(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		mr, _ := r.MultipartReader()
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
			if part.FormName() == "file" {
				seen = part.FileName()
			}
			_, _ = io.Copy(io.Discard, part)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:       "whisper-1",
		Audio:       bytes.NewReader([]byte("a")),
		AudioFormat: "audio/wav",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if seen != "audio.wav" {
		t.Errorf("filename = %q, want audio.wav", seen)
	}
}

func TestTranscribe_FilenameExplicit(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		mr, _ := r.MultipartReader()
		for {
			part, err := mr.NextPart()
			if err != nil {
				break
			}
			if part.FormName() == "file" {
				seen = part.FileName()
			}
			_, _ = io.Copy(io.Discard, part)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:    "whisper-1",
		Audio:    bytes.NewReader([]byte("a")),
		Filename: "voice.m4a",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if seen != "voice.m4a" {
		t.Errorf("filename = %q", seen)
	}
}

func TestTranscribe_AuthorizationHeader(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q", seen)
	}
}

func TestTranscribe_ContentTypeIsMultipart(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Content-Type")
	}))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	mediaType, _, err := mime.ParseMediaType(seen)
	if err != nil {
		t.Fatalf("ParseMediaType: %v", err)
	}
	if mediaType != "multipart/form-data" {
		t.Errorf("media type = %q, want multipart/form-data", mediaType)
	}
}

func TestTranscribe_PlainTextResponse(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "raw transcribed text")
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:          "whisper-1",
		Audio:          bytes.NewReader([]byte("a")),
		ResponseFormat: "text",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var segs []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segs = append(segs, s)
	}
	if len(segs) != 1 || segs[0].Text != "raw transcribed text" {
		t.Errorf("segs = %+v", segs)
	}
	if !segs[0].Final {
		t.Errorf("Final = false")
	}
}

func TestTranscribe_PlainJSON(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, `{"text":"basic only"}`, nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:          "whisper-1",
		Audio:          bytes.NewReader([]byte("a")),
		ResponseFormat: "json",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var segs []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segs = append(segs, s)
	}
	if len(segs) != 1 || segs[0].Text != "basic only" {
		t.Errorf("segs = %+v", segs)
	}
	if len(segs[0].Words) != 0 {
		t.Errorf("Words should be empty for json format")
	}
}

func TestTranscribe_SRTResponseFormat(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "1\n00:00:00,000 --> 00:00:01,000\nhello\n")
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:          "whisper-1",
		Audio:          bytes.NewReader([]byte("a")),
		ResponseFormat: "srt",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var segs []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segs = append(segs, s)
	}
	if !strings.Contains(segs[0].Text, "hello") {
		t.Errorf("Text = %q", segs[0].Text)
	}
}

func TestTranscribe_4xx_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) || upstream.StatusCode != 400 {
		t.Fatalf("err = %v, want 400 ErrUpstream", err)
	}
}

func TestTranscribe_500_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) || upstream.StatusCode != 500 {
		t.Fatalf("err = %v, want 500 ErrUpstream", err)
	}
}

func TestTranscribe_NilAudio(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{Model: "whisper-1"})
	if err == nil {
		t.Fatalf("expected error for nil audio")
	}
}

func TestTranscribe_TemperatureForwarded(t *testing.T) {
	var seenTemp string
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		mr, _ := r.MultipartReader()
		for {
			part, err := mr.NextPart()
			if err != nil {
				break
			}
			data, _ := io.ReadAll(part)
			if part.FormName() == "temperature" {
				seenTemp = string(data)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	temp := 0.5
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:       "whisper-1",
		Audio:       bytes.NewReader([]byte("a")),
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if seenTemp != "0.5" {
		t.Errorf("temperature = %q, want 0.5", seenTemp)
	}
}

func TestTranscribe_StreamFlagIgnored(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:  "whisper-1",
		Audio:  bytes.NewReader([]byte("a")),
		Stream: true,
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	count := 0
	for range stream.Segments() {
		count++
	}
	if count != 1 {
		t.Errorf("segment count = %d, want 1 (streaming ignored)", count)
	}
}
