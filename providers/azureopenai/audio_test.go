package azureopenai_test

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
	"github.com/elloloop/llmrouter/providers/azureopenai"
)

func fakeAudioServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

func newAzureAudioProvider(t *testing.T, baseURL string) *azureopenai.Provider {
	t.Helper()
	p, err := azureopenai.New(
		llmrouter.WithAPIKey(testKey),
		llmrouter.WithBaseURL(baseURL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func newAzureAudioProviderAAD(t *testing.T, baseURL string, tok string) *azureopenai.Provider {
	t.Helper()
	src := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) { return tok, nil })
	p, err := azureopenai.New(
		azureopenai.WithAADToken(src),
		llmrouter.WithBaseURL(baseURL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// ---------- Speak ----------

func speechBytes() []byte { return []byte("AZURE-AUDIO-DATA") }

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

func TestAzureSpeak_NonStreaming(t *testing.T) {
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	var got []byte
	for chunk := range stream.Chunks() {
		got = append(got, chunk.Data...)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if !bytes.Equal(got, speechBytes()) {
		t.Errorf("got=%q want=%q", got, speechBytes())
	}
}

func TestAzureSpeak_URLPath(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Path
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	want := "/openai/deployments/" + testDeployment + "/audio/speech"
	if seen != want {
		t.Errorf("path = %q, want %q", seen, want)
	}
}

func TestAzureSpeak_APIVersionQuery(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Query().Get("api-version")
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	if seen != testAPIVersion {
		t.Errorf("api-version = %q", seen)
	}
}

func TestAzureSpeak_APIKeyHeader(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	if seenKey != testKey {
		t.Errorf("api-key = %q", seenKey)
	}
	if seenAuth != "" {
		t.Errorf("Authorization set for api-key auth: %q", seenAuth)
	}
}

func TestAzureSpeak_AADBearer(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAzureAudioProviderAAD(t, srv.URL, "aad-token")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	if seenKey != "" {
		t.Errorf("api-key set for AAD auth: %q", seenKey)
	}
	if seenAuth != "Bearer aad-token" {
		t.Errorf("Authorization = %q", seenAuth)
	}
}

func TestAzureSpeak_FormatDefault(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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

func TestAzureSpeak_FormatExplicit(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Format: "flac"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["response_format"] != "flac" {
		t.Errorf("response_format = %v", m["response_format"])
	}
}

func TestAzureSpeak_SpeedPointer(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	speed := 0.75
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi", Speed: &speed})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["speed"].(float64) != 0.75 {
		t.Errorf("speed = %v", m["speed"])
	}
}

func TestAzureSpeak_RawOverlay(t *testing.T) {
	var body []byte
	srv := fakeAudioServer(t, speechHandler(t, speechBytes(), func(t *testing.T, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw := json.RawMessage(`{"vendor":"y","model":"old","input":"old"}`)
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "new-input", Raw: raw})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	for range stream.Chunks() {
	}
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	if m["vendor"] != "y" {
		t.Errorf("vendor extra dropped")
	}
	if m["model"] != "tts-1" || m["input"] != "new-input" {
		t.Errorf("overlay failed: %v", m)
	}
}

func TestAzureSpeak_Streaming_Chunks(t *testing.T) {
	big := bytes.Repeat([]byte("B"), 20*1024)
	srv := fakeAudioServer(t, speechHandler(t, big, nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	if chunks < 2 {
		t.Errorf("chunks = %d, want >=2", chunks)
	}
	if !bytes.Equal(got, big) {
		t.Errorf("body mismatch")
	}
}

func TestAzureSpeak_ContentTypePropagated(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("wav"))
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if stream.ContentType != "audio/wav" {
		t.Errorf("ContentType = %q", stream.ContentType)
	}
	for range stream.Chunks() {
	}
}

func TestAzureSpeak_4xx_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad"))
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Speak(ctx, llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) || upstream.StatusCode != 400 {
		t.Fatalf("err = %v, want 400 ErrUpstream", err)
	}
	if upstream.Provider != "azureopenai" {
		t.Errorf("Provider = %q", upstream.Provider)
	}
}

// ---------- Transcribe ----------

func transcribeVerboseJSON() string {
	return `{
		"text":"azure transcribe",
		"segments":[{"text":"azure transcribe","start":0,"end":1.0}],
		"words":[{"word":"azure","start":0,"end":0.5},{"word":"transcribe","start":0.5,"end":1.0}]
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

func TestAzureTranscribe_HappyPath(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("AUDIO")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var segs []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segs = append(segs, s)
	}
	if len(segs) != 1 {
		t.Fatalf("segs = %d", len(segs))
	}
	if segs[0].Text != "azure transcribe" {
		t.Errorf("text = %q", segs[0].Text)
	}
	if len(segs[0].Words) != 2 {
		t.Errorf("words = %d", len(segs[0].Words))
	}
	if segs[0].End != time.Second {
		t.Errorf("end = %v, want 1s", segs[0].End)
	}
}

func TestAzureTranscribe_URLPath(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Path
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	want := "/openai/deployments/" + testDeployment + "/audio/transcriptions"
	if seen != want {
		t.Errorf("path = %q, want %q", seen, want)
	}
}

func TestAzureTranscribe_APIVersionQuery(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Query().Get("api-version")
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	if seen != testAPIVersion {
		t.Errorf("api-version = %q", seen)
	}
}

func TestAzureTranscribe_APIKeyHeader(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	if seenKey != testKey {
		t.Errorf("api-key = %q", seenKey)
	}
	if seenAuth != "" {
		t.Errorf("Authorization unexpectedly set: %q", seenAuth)
	}
}

func TestAzureTranscribe_AADBearer(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	}))
	defer srv.Close()
	p := newAzureAudioProviderAAD(t, srv.URL, "tok-2")
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
	if seenKey != "" {
		t.Errorf("api-key set for AAD: %q", seenKey)
	}
	if seenAuth != "Bearer tok-2" {
		t.Errorf("Authorization = %q", seenAuth)
	}
}

func TestAzureTranscribe_MultipartFields(t *testing.T) {
	var got struct {
		file, model, lang, prompt, fmt_ string
		filename                        string
	}
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
				got.file = string(data)
				got.filename = part.FileName()
			case "model":
				got.model = string(data)
			case "language":
				got.lang = string(data)
			case "prompt":
				got.prompt = string(data)
			case "response_format":
				got.fmt_ = string(data)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:    "whisper-1",
		Audio:    bytes.NewReader([]byte("HELLO")),
		Language: "fr",
		Prompt:   "ctx",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if got.file != "HELLO" {
		t.Errorf("file = %q", got.file)
	}
	if got.model != "whisper-1" {
		t.Errorf("model = %q", got.model)
	}
	if got.lang != "fr" {
		t.Errorf("language = %q", got.lang)
	}
	if got.prompt != "ctx" {
		t.Errorf("prompt = %q", got.prompt)
	}
	if got.fmt_ != "verbose_json" {
		t.Errorf("response_format = %q", got.fmt_)
	}
	if got.filename == "" {
		t.Errorf("filename empty")
	}
}

func TestAzureTranscribe_ContentTypeMultipart(t *testing.T) {
	var seen string
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Content-Type")
	}))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
		t.Errorf("mediaType = %q", mediaType)
	}
}

func TestAzureTranscribe_PlainTextFormat(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "azure text")
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	if segs[0].Text != "azure text" {
		t.Errorf("text = %q", segs[0].Text)
	}
}

func TestAzureTranscribe_PlainJSON(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, `{"text":"basic"}`, nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
	if segs[0].Text != "basic" {
		t.Errorf("text = %q", segs[0].Text)
	}
}

func TestAzureTranscribe_4xx_ErrUpstream(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad"))
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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

func TestAzureTranscribe_NilAudio(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{Model: "whisper-1"})
	if err == nil {
		t.Fatalf("expected error for nil audio")
	}
}

func TestAzureTranscribe_TemperatureForwarded(t *testing.T) {
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
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	temp := 0.25
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
	if seenTemp != "0.25" {
		t.Errorf("temperature = %q", seenTemp)
	}
}

func TestAzureTranscribe_ContextCancelled(t *testing.T) {
	srv := fakeAudioServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, transcribeVerboseJSON())
	})
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	if err == nil {
		t.Fatalf("expected error from cancelled context")
	}
}

func TestAzureTranscribe_RawSegmentPreserved(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("a")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var seg llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		seg = s
	}
	if !strings.Contains(string(seg.Raw), "azure transcribe") {
		t.Errorf("Raw missing payload: %s", seg.Raw)
	}
}

func TestAzureTranscribe_StreamFlagIgnored(t *testing.T) {
	srv := fakeAudioServer(t, transcribeHandler(t, http.StatusOK, transcribeVerboseJSON(), nil))
	defer srv.Close()
	p := newAzureAudioProvider(t, srv.URL)
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
		t.Errorf("segments = %d, want 1", count)
	}
}
