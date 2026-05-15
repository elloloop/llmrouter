package vertex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// --- pure helpers ----------------------------------------------------------

func TestVertex_BuildSpeakConfig_NoVoice(t *testing.T) {
	cfg := buildSpeakConfig(llmrouter.SpeechRequest{Input: "hi"})
	if len(cfg.ResponseModalities) != 1 || cfg.ResponseModalities[0] != "AUDIO" {
		t.Errorf("ResponseModalities = %v", cfg.ResponseModalities)
	}
	if cfg.SpeechConfig != nil {
		t.Errorf("SpeechConfig should be nil when no voice supplied")
	}
}

func TestVertex_BuildSpeakConfig_WithVoice(t *testing.T) {
	cfg := buildSpeakConfig(llmrouter.SpeechRequest{Voice: "puck"})
	if cfg.SpeechConfig == nil ||
		cfg.SpeechConfig.VoiceConfig == nil ||
		cfg.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig == nil {
		t.Fatalf("SpeechConfig chain not populated: %+v", cfg.SpeechConfig)
	}
	if got := cfg.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig.VoiceName; got != "puck" {
		t.Errorf("VoiceName = %q", got)
	}
}

func TestVertex_SampleRateFromMIME(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"audio/L16;rate=24000", 24000},
		{"audio/L16;rate=16000", 16000},
		{"audio/L16", 24000},
		{"", 24000},
		{"audio/L16;rate=NOPE", 24000},
		{"audio/L16; rate=44100", 44100},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := sampleRateFromMIME(tc.in); got != tc.want {
				t.Errorf("sampleRateFromMIME(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

func TestVertex_WrapPCMAsWAV_HeaderShape(t *testing.T) {
	pcm := []byte{1, 2, 3, 4}
	out := wrapPCMAsWAV(pcm, 24000, 1, 16)
	if len(out) != 44+len(pcm) {
		t.Fatalf("len = %d, want %d", len(out), 44+len(pcm))
	}
	if string(out[0:4]) != "RIFF" {
		t.Errorf("missing RIFF marker")
	}
	if string(out[8:12]) != "WAVE" {
		t.Errorf("missing WAVE marker")
	}
	if rate := binary.LittleEndian.Uint32(out[24:28]); rate != 24000 {
		t.Errorf("rate = %d", rate)
	}
}

func TestVertex_FinalizeAudio_WAVPath(t *testing.T) {
	ct, data := finalizeAudio("wav", "audio/L16;rate=24000", []byte("xy"))
	if ct != "audio/wav" {
		t.Errorf("ct = %q", ct)
	}
	if len(data) <= 44 || string(data[0:4]) != "RIFF" {
		t.Errorf("expected WAV header")
	}
}

func TestVertex_FinalizeAudio_Passthrough(t *testing.T) {
	ct, data := finalizeAudio("", "audio/L16;rate=24000", []byte("xy"))
	if ct != "audio/L16;rate=24000" {
		t.Errorf("ct = %q", ct)
	}
	if !bytes.Equal(data, []byte("xy")) {
		t.Errorf("data mutated")
	}
}

func TestVertex_FinalizeAudio_FallbackContentType(t *testing.T) {
	ct, _ := finalizeAudio("", "", []byte("x"))
	if ct != "application/octet-stream" {
		t.Errorf("ct = %q", ct)
	}
}

func TestVertex_TranscribePrompt(t *testing.T) {
	cases := []struct {
		name string
		req  llmrouter.TranscribeRequest
		want string
	}{
		{"bare", llmrouter.TranscribeRequest{}, "Transcribe this audio."},
		{"lang", llmrouter.TranscribeRequest{Language: "fr"}, "Transcribe this fr audio."},
		{"prompt", llmrouter.TranscribeRequest{Prompt: "Names: A."}, "Transcribe this audio. Names: A."},
		{"both", llmrouter.TranscribeRequest{Language: "es", Prompt: "ctx"}, "Transcribe this es audio. ctx"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := transcribePrompt(tc.req); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestVertex_ExtractFirstInlineAudio_Empty(t *testing.T) {
	_, _, err := extractFirstInlineAudio(&genai.GenerateContentResponse{})
	if err == nil {
		t.Fatal("expected error on empty response")
	}
}

func TestVertex_ExtractFirstInlineAudio_Nil(t *testing.T) {
	_, _, err := extractFirstInlineAudio(nil)
	if err == nil {
		t.Fatal("expected error on nil response")
	}
}

func TestVertex_ExtractFirstInlineAudio_FoundsFirstBlob(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "noise"},
					{InlineData: &genai.Blob{MIMEType: "audio/L16;rate=24000", Data: []byte("AB")}},
				},
			},
		}},
	}
	mime, data, err := extractFirstInlineAudio(resp)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if mime != "audio/L16;rate=24000" {
		t.Errorf("mime = %q", mime)
	}
	if !bytes.Equal(data, []byte("AB")) {
		t.Errorf("data = %q", data)
	}
}

func TestVertex_ExtractTranscriptText(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "hello "},
					{Text: "world"},
				},
			},
		}},
	}
	got, err := extractTranscriptText(resp)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got != "hello world" {
		t.Errorf("text = %q", got)
	}
}

func TestVertex_ExtractTranscriptText_NoCandidates(t *testing.T) {
	_, err := extractTranscriptText(&genai.GenerateContentResponse{})
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- input validation -----------------------------------------------------

func TestVertex_Speak_RequiresModel(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "x"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestVertex_Speak_RequiresInput(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestVertex_Transcribe_RequiresModel(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: bytes.NewReader([]byte("a")),
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestVertex_Transcribe_RequiresAudio(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- integration via fake genai server -----------------------------------

// newFakeAudioServer is a thin wrapper around httptest that records each
// inbound POST body for assertions.
func newFakeAudioServer(t *testing.T, respBody string, status int) (*httptest.Server, *[]byte) {
	t.Helper()
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		captured = b
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, respBody)
	}))
	return srv, &captured
}

// newAudioProvider returns a Provider whose genai client points to srv.
func newAudioProvider(t *testing.T, srv *httptest.Server) *Provider {
	t.Helper()
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:     genai.BackendVertexAI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatalf("genai client: %v", err)
	}
	return &Provider{cfg: &llmrouter.Config{}, client: client}
}

func TestVertex_Speak_UpstreamError(t *testing.T) {
	srv, _ := newFakeAudioServer(t, `{"error":{"code":403,"message":"nope"}}`, http.StatusForbidden)
	defer srv.Close()
	p := newAudioProvider(t, srv)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if up.Provider != providerName {
		t.Errorf("Provider = %q", up.Provider)
	}
}

func TestVertex_Transcribe_UpstreamError(t *testing.T) {
	srv, _ := newFakeAudioServer(t, `{"error":{"code":500,"message":"boom"}}`, http.StatusInternalServerError)
	defer srv.Close()
	p := newAudioProvider(t, srv)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "m",
		Audio: bytes.NewReader([]byte("audio")),
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
}

func TestVertex_Speak_BodyContainsAudioModality(t *testing.T) {
	// The genai SDK encodes ResponseModalities and SpeechConfig into the
	// outgoing JSON body. We only assert the byte stream contains the
	// expected markers — the exact key path differs between mldev and
	// vertex encoders.
	srv, captured := newFakeAudioServer(t, `{}`, http.StatusOK)
	defer srv.Close()
	p := newAudioProvider(t, srv)
	_, _ = p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hello",
		Voice: "kore",
	})
	body := string(*captured)
	if !strings.Contains(body, "AUDIO") {
		t.Errorf("body missing AUDIO modality: %s", body)
	}
	if !strings.Contains(body, "kore") {
		t.Errorf("body missing voice name: %s", body)
	}
}

func TestVertex_Transcribe_BodyContainsInlineAudio(t *testing.T) {
	srv, captured := newFakeAudioServer(t, `{}`, http.StatusOK)
	defer srv.Close()
	p := newAudioProvider(t, srv)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model:       "m",
		Audio:       bytes.NewReader([]byte("BINARY")),
		AudioFormat: "audio/wav",
		Prompt:      "Names: Alice",
	})
	body := string(*captured)
	// Inline base64 of "BINARY"
	encoded := base64.StdEncoding.EncodeToString([]byte("BINARY"))
	if !strings.Contains(body, encoded) {
		t.Errorf("body missing inline audio base64: %s", body)
	}
	if !strings.Contains(body, "audio/wav") {
		t.Errorf("body missing audio/wav MIME: %s", body)
	}
	if !strings.Contains(body, "Names: Alice") {
		t.Errorf("body missing prompt: %s", body)
	}
}
