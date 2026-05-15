package elevenlabs_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
)

// capturedSTTRequest holds the parsed multipart form + headers from an
// inbound /v1/speech-to-text request for assertion convenience.
type capturedSTTRequest struct {
	method      string
	path        string
	apiKey      string
	contentType string
	fields      map[string]string
	fileBytes   []byte
	filename    string
}

// fakeSTTServer stands up an httptest server that parses the multipart
// body, records what it saw, and writes the given status + body.
func fakeSTTServer(t *testing.T, status int, respBody string) (*httptest.Server, *capturedSTTRequest) {
	t.Helper()
	cap := &capturedSTTRequest{fields: map[string]string{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.apiKey = r.Header.Get("xi-api-key")
		cap.contentType = r.Header.Get("Content-Type")

		mediaType, params, err := mime.ParseMediaType(cap.contentType)
		if err == nil && strings.HasPrefix(mediaType, "multipart/") {
			mr := multipart.NewReader(r.Body, params["boundary"])
			for {
				part, err := mr.NextPart()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Errorf("NextPart: %v", err)
					break
				}
				buf, _ := io.ReadAll(part)
				if part.FileName() != "" {
					cap.fileBytes = buf
					cap.filename = part.FileName()
				} else {
					cap.fields[part.FormName()] = string(buf)
				}
				_ = part.Close()
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(respBody))
	}))
	return srv, cap
}

func newSTTProvider(t *testing.T, srvURL string) *elevenlabs.Provider {
	t.Helper()
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srvURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func sampleAudio() io.Reader {
	return bytes.NewReader([]byte("RIFFfake-wav-bytes"))
}

func TestTranscribe_SuccessReturnsFinalSegment(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusOK, `{"text":"hello world","language_code":"en"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)

	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio:       sampleAudio(),
		AudioFormat: "audio/wav",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	var segs []llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		segs = append(segs, s)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v, want nil", err)
	}
	if len(segs) != 1 {
		t.Fatalf("segment count = %d, want 1", len(segs))
	}
	if segs[0].Text != "hello world" {
		t.Errorf("Text = %q, want hello world", segs[0].Text)
	}
	if !segs[0].Final {
		t.Errorf("Final = false, want true")
	}
}

func TestTranscribe_HitsSpeechToTextPath(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: sampleAudio(),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.path != "/v1/speech-to-text" {
		t.Errorf("path = %q, want /v1/speech-to-text", cap.path)
	}
	if cap.method != http.MethodPost {
		t.Errorf("method = %q, want POST", cap.method)
	}
}

func TestTranscribe_ContentTypeIsMultipart(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: sampleAudio(),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if !strings.HasPrefix(cap.contentType, "multipart/form-data") {
		t.Errorf("Content-Type = %q, want multipart prefix", cap.contentType)
	}
}

func TestTranscribe_AuthUsesXIAPIKey(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.apiKey != "test-key" {
		t.Errorf("xi-api-key = %q, want test-key", cap.apiKey)
	}
}

func TestTranscribe_DefaultModelIsScribeV1(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.fields["model_id"] != "scribe_v1" {
		t.Errorf("model_id = %q, want scribe_v1", cap.fields["model_id"])
	}
}

func TestTranscribe_ExplicitModelPropagated(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: sampleAudio(),
		Model: "scribe_v1_experimental",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.fields["model_id"] != "scribe_v1_experimental" {
		t.Errorf("model_id = %q, want scribe_v1_experimental", cap.fields["model_id"])
	}
}

func TestTranscribe_LanguageCodeForwardedWhenSet(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio:    sampleAudio(),
		Language: "fr",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.fields["language_code"] != "fr" {
		t.Errorf("language_code = %q, want fr", cap.fields["language_code"])
	}
}

func TestTranscribe_LanguageCodeOmittedWhenEmpty(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if _, ok := cap.fields["language_code"]; ok {
		t.Errorf("language_code unexpectedly present: %q", cap.fields["language_code"])
	}
}

func TestTranscribe_TimestampGranularityIsWord(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if got := cap.fields["timestamps_granularity"]; got != "word" {
		t.Errorf("timestamps_granularity = %q, want word", got)
	}
}

func TestTranscribe_TagAudioEventsTrue(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if got := cap.fields["tag_audio_events"]; got != "true" {
		t.Errorf("tag_audio_events = %q, want true", got)
	}
}

func TestTranscribe_DiarizeFromRawFalse(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: sampleAudio(),
		Raw:   json.RawMessage(`{"diarize":false}`),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if got := cap.fields["diarize"]; got != "false" {
		t.Errorf("diarize = %q, want false", got)
	}
}

func TestTranscribe_DiarizeFromRawTrue(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: sampleAudio(),
		Raw:   json.RawMessage(`{"diarize":true}`),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if got := cap.fields["diarize"]; got != "true" {
		t.Errorf("diarize = %q, want true", got)
	}
}

func TestTranscribe_DiarizeOmittedWhenNotInRaw(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if _, ok := cap.fields["diarize"]; ok {
		t.Errorf("diarize unexpectedly present: %q", cap.fields["diarize"])
	}
}

func TestTranscribe_WordsConvertedToTranscriptWords(t *testing.T) {
	body := `{
		"text":"hello world",
		"language_code":"en",
		"words":[
			{"text":"hello","type":"word","start":0.01,"end":0.30,"speaker_id":"speaker_1"},
			{"text":" ","type":"spacing","start":0.30,"end":0.32},
			{"text":"world","type":"word","start":0.32,"end":0.75,"speaker_id":"speaker_1"}
		]
	}`
	srv, _ := fakeSTTServer(t, http.StatusOK, body)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	var seg llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		seg = s
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}

	if len(seg.Words) != 2 {
		t.Fatalf("Words count = %d, want 2 (spacing entries filtered)", len(seg.Words))
	}
	if seg.Words[0].Word != "hello" {
		t.Errorf("Words[0].Word = %q, want hello", seg.Words[0].Word)
	}
	wantStart := time.Duration(0.01 * float64(time.Second))
	wantEnd := time.Duration(0.30 * float64(time.Second))
	if seg.Words[0].Start != wantStart {
		t.Errorf("Words[0].Start = %v, want %v", seg.Words[0].Start, wantStart)
	}
	if seg.Words[0].End != wantEnd {
		t.Errorf("Words[0].End = %v, want %v", seg.Words[0].End, wantEnd)
	}
	// Segment timing taken from first/last filtered word.
	if seg.Start != wantStart {
		t.Errorf("seg.Start = %v, want %v", seg.Start, wantStart)
	}
	if seg.End != time.Duration(0.75*float64(time.Second)) {
		t.Errorf("seg.End = %v, want 750ms", seg.End)
	}
}

func TestTranscribe_NoWordsLeavesWordsNil(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusOK, `{"text":"plain","language_code":"en"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	var seg llmrouter.TranscriptSegment
	for s := range stream.Segments() {
		seg = s
	}
	if len(seg.Words) != 0 {
		t.Errorf("Words = %v, want empty", seg.Words)
	}
	if seg.Text != "plain" {
		t.Errorf("Text = %q, want plain", seg.Text)
	}
}

func TestTranscribe_FilenameFromAudioFormat(t *testing.T) {
	cases := []struct {
		format string
		ext    string
	}{
		{"audio/mpeg", ".mp3"},
		{"audio/wav", ".wav"},
		{"audio/webm", ".webm"},
		{"audio/flac", ".flac"},
		{"audio/m4a", ".m4a"},
		{"", ".mp3"},
		{"weird/unknown", ".mp3"},
	}
	for _, c := range cases {
		t.Run("fmt="+c.format, func(t *testing.T) {
			srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
			defer srv.Close()

			p := newSTTProvider(t, srv.URL)
			stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
				Audio:       sampleAudio(),
				AudioFormat: c.format,
			})
			if err != nil {
				t.Fatalf("Transcribe: %v", err)
			}
			for range stream.Segments() {
			}
			if !strings.HasSuffix(cap.filename, c.ext) {
				t.Errorf("filename %q does not end with %q", cap.filename, c.ext)
			}
		})
	}
}

func TestTranscribe_ExplicitFilenameOverridesFormat(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio:       sampleAudio(),
		AudioFormat: "audio/wav",
		Filename:    "recording.flac",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.filename != "recording.flac" {
		t.Errorf("filename = %q, want recording.flac", cap.filename)
	}
}

func TestTranscribe_AudioBodyForwarded(t *testing.T) {
	srv, cap := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: bytes.NewReader([]byte("PAYLOAD-ABC")),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if string(cap.fileBytes) != "PAYLOAD-ABC" {
		t.Errorf("file bytes = %q, want PAYLOAD-ABC", cap.fileBytes)
	}
}

func TestTranscribe_4xxReturnsErrUpstream(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusUnauthorized, `{"detail":"bad key"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %T, want *ErrUpstream", err)
	}
	if upErr.Provider != "elevenlabs" {
		t.Errorf("Provider = %q, want elevenlabs", upErr.Provider)
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", upErr.StatusCode)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("Body = %q, want substring 'bad key'", upErr.Body)
	}
}

func TestTranscribe_5xxReturnsErrUpstream(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusBadGateway, "down")
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if upErr.StatusCode != http.StatusBadGateway {
		t.Errorf("StatusCode = %d, want 502", upErr.StatusCode)
	}
}

func TestTranscribe_NilAudioReaderRejected(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{})
	if err == nil {
		t.Fatal("expected error for nil Audio reader, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if errors.As(err, &upErr) {
		t.Fatalf("validation error wrongly wrapped as ErrUpstream: %v", err)
	}
}

func TestTranscribe_StreamFlagIgnoredButAccepted(t *testing.T) {
	// req.Stream = true must not cause an error — it is documented as
	// accepted-but-ignored. Behaviour matches the non-streaming path.
	srv, _ := fakeSTTServer(t, http.StatusOK, `{"text":"ok"}`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio:  sampleAudio(),
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
		t.Errorf("segment count = %d, want 1 (stream flag is ignored)", count)
	}
}

func TestTranscribe_HTTPClientErrorBubbles(t *testing.T) {
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithBaseURL("http://127.0.0.1:1"),
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err == nil {
		t.Fatal("expected transport error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if errors.As(err, &upErr) {
		t.Fatalf("transport error wrongly wrapped as ErrUpstream: %v", err)
	}
}

func TestTranscribe_MalformedJSONReturnsError(t *testing.T) {
	srv, _ := fakeSTTServer(t, http.StatusOK, `not-json-at-all`)
	defer srv.Close()

	p := newSTTProvider(t, srv.URL)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Audio: sampleAudio()})
	if err == nil {
		t.Fatal("expected decode error, got nil")
	}
}
