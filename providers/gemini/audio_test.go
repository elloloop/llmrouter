package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// audioRecorder is the analogue of embedRecorder for audio tests.
type audioRecorder struct {
	srv      *httptest.Server
	urlPath  string
	body     []byte
	headers  http.Header
	respCode int
	respBody string
}

func newAudioRecorder(t *testing.T) *audioRecorder {
	t.Helper()
	ar := &audioRecorder{respCode: http.StatusOK}
	ar.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		ar.body = b
		ar.urlPath = r.URL.Path
		ar.headers = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(ar.respCode)
		_, _ = io.WriteString(w, ar.respBody)
	}))
	return ar
}

func (a *audioRecorder) close() { a.srv.Close() }

func (a *audioRecorder) newProvider(t *testing.T) *Provider {
	t.Helper()
	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(a.srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// makeSpeakResponse encodes a synthetic generateContent reply containing
// one inline audio part.
func makeSpeakResponse(mime string, audio []byte) string {
	type inline struct {
		MIMEType string `json:"mimeType"`
		Data     string `json:"data"`
	}
	type part struct {
		InlineData inline `json:"inlineData"`
	}
	type content struct {
		Parts []part `json:"parts"`
	}
	type cand struct {
		Content content `json:"content"`
	}
	resp := map[string]any{
		"candidates": []cand{{Content: content{Parts: []part{{
			InlineData: inline{MIMEType: mime, Data: base64.StdEncoding.EncodeToString(audio)},
		}}}}},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// collectAudio drains all chunks from a stream into one byte slice.
func collectAudio(s *llmrouter.AudioStream) ([]byte, error) {
	var buf bytes.Buffer
	for c := range s.Chunks() {
		buf.Write(c.Data)
	}
	return buf.Bytes(), s.Err()
}

// --- Speak / TTS -----------------------------------------------------------

func TestSpeak_DecodesBase64Audio(t *testing.T) {
	want := []byte("rawpcmbytes")
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", want)
	p := r.newProvider(t)
	s, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "gemini-2.5-flash-tts",
		Input: "hello",
		Voice: "kore",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	got, err := collectAudio(s)
	if err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("audio bytes mismatch: got %q want %q", got, want)
	}
}

func TestSpeak_PassesContentTypeWhenNotWAV(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", []byte("x"))
	p := r.newProvider(t)
	s, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	_, _ = collectAudio(s)
	if s.ContentType != "audio/L16;rate=24000" {
		t.Errorf("ContentType = %q", s.ContentType)
	}
}

func TestSpeak_WAVWrappingEnabled(t *testing.T) {
	pcm := []byte("\x01\x02\x03\x04")
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", pcm)
	p := r.newProvider(t)
	s, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model:  "m",
		Input:  "hi",
		Format: "wav",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	got, err := collectAudio(s)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s.ContentType != "audio/wav" {
		t.Errorf("ContentType = %q, want audio/wav", s.ContentType)
	}
	if len(got) < 44 {
		t.Fatalf("wav payload too short: %d", len(got))
	}
	if string(got[0:4]) != "RIFF" || string(got[8:12]) != "WAVE" {
		t.Errorf("WAV header missing: %q %q", got[0:4], got[8:12])
	}
	if !bytes.Equal(got[44:], pcm) {
		t.Errorf("PCM bytes not appended after header")
	}
}

func TestSpeak_WAVWrappingUsesSampleRateFromMIME(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=16000", []byte("ab"))
	p := r.newProvider(t)
	s, _ := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model:  "m",
		Input:  "hi",
		Format: "wav",
	})
	got, _ := collectAudio(s)
	rate := binary.LittleEndian.Uint32(got[24:28])
	if rate != 16000 {
		t.Errorf("sampleRate = %d, want 16000", rate)
	}
}

func TestSpeak_WAVDefaultsTo24kWhenMIMEMissingRate(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16", []byte("ab"))
	p := r.newProvider(t)
	s, _ := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model:  "m",
		Input:  "hi",
		Format: "wav",
	})
	got, _ := collectAudio(s)
	rate := binary.LittleEndian.Uint32(got[24:28])
	if rate != 24000 {
		t.Errorf("default sampleRate = %d, want 24000", rate)
	}
}

func TestSpeak_BodyShape(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", []byte("x"))
	p := r.newProvider(t)
	_, _ = p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "gemini-2.5-flash-tts",
		Input: "hello",
		Voice: "kore",
	})
	var body map[string]any
	if err := json.Unmarshal(r.body, &body); err != nil {
		t.Fatalf("body json: %v", err)
	}
	gen, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing: %v", body)
	}
	mods, _ := gen["responseModalities"].([]any)
	if len(mods) != 1 || mods[0] != "AUDIO" {
		t.Errorf("responseModalities = %v", mods)
	}
	speech, ok := gen["speechConfig"].(map[string]any)
	if !ok {
		t.Fatalf("speechConfig missing")
	}
	voiceCfg, _ := speech["voiceConfig"].(map[string]any)
	prebuilt, _ := voiceCfg["prebuiltVoiceConfig"].(map[string]any)
	if prebuilt["voiceName"] != "kore" {
		t.Errorf("voiceName = %v", prebuilt["voiceName"])
	}
}

func TestSpeak_HitsGenerateContentPath(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", []byte("x"))
	p := r.newProvider(t)
	_, _ = p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	if !strings.HasSuffix(r.urlPath, ":generateContent") {
		t.Errorf("path = %q, want :generateContent suffix", r.urlPath)
	}
}

func TestSpeak_SetsAuthHeader(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", []byte("x"))
	p := r.newProvider(t)
	_, _ = p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	if got := r.headers.Get(apiKeyHeader); got != "test-key" {
		t.Errorf("%s = %q", apiKeyHeader, got)
	}
}

func TestSpeak_StreamFlagIgnored(t *testing.T) {
	// Stream=true is documented as accepted-but-ignored — the call should
	// succeed and deliver the entire audio in one chunk.
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeSpeakResponse("audio/L16;rate=24000", []byte("hello"))
	p := r.newProvider(t)
	s, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model:  "m",
		Input:  "hi",
		Stream: true,
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	count := 0
	for range s.Chunks() {
		count++
	}
	if count != 1 {
		t.Errorf("chunks = %d, want 1", count)
	}
}

func TestSpeak_UpstreamError(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respCode = http.StatusForbidden
	r.respBody = `{"error":"forbidden"}`
	p := r.newProvider(t)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if up.StatusCode != 403 {
		t.Errorf("StatusCode = %d", up.StatusCode)
	}
}

func TestSpeak_NoAudioInResponse(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = `{"candidates":[{"content":{"parts":[]}}]}`
	p := r.newProvider(t)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Model: "m",
		Input: "hi",
	})
	if err == nil {
		t.Fatal("expected error when response has no audio")
	}
}

func TestSpeak_RequiresInput(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Model: "m"})
	if err == nil {
		t.Fatal("expected error on empty input")
	}
}

func TestSpeak_RequiresModel(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err == nil {
		t.Fatal("expected error on empty model")
	}
}

// --- Transcribe / STT -----------------------------------------------------

// makeTranscribeResponse encodes a synthetic generateContent reply
// containing transcribed text.
func makeTranscribeResponse(text string) string {
	resp := map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"parts": []map[string]any{{"text": text}},
			},
		}},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

func collectTranscript(s *llmrouter.TranscriptStream) ([]llmrouter.TranscriptSegment, error) {
	var out []llmrouter.TranscriptSegment
	for seg := range s.Segments() {
		out = append(out, seg)
	}
	return out, s.Err()
}

func TestTranscribe_SingleFinalSegment(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("hello world")
	p := r.newProvider(t)
	s, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model:       "gemini-2.0-flash",
		Audio:       bytes.NewReader([]byte("FAKEAUDIO")),
		AudioFormat: "audio/mpeg",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	segs, err := collectTranscript(s)
	if err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if len(segs) != 1 {
		t.Fatalf("segments = %d, want 1", len(segs))
	}
	if !segs[0].Final {
		t.Errorf("segment should be Final=true")
	}
	if segs[0].Text != "hello world" {
		t.Errorf("Text = %q", segs[0].Text)
	}
}

func TestTranscribe_BodyContainsInlineAudio(t *testing.T) {
	audio := []byte("BINARYAUDIO")
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("ok")
	p := r.newProvider(t)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model:       "m",
		Audio:       bytes.NewReader(audio),
		AudioFormat: "audio/wav",
	})
	var body struct {
		Contents []struct {
			Parts []struct {
				InlineData *struct {
					MIMEType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData,omitempty"`
				Text string `json:"text,omitempty"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(r.body, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(body.Contents) != 1 || len(body.Contents[0].Parts) != 2 {
		t.Fatalf("unexpected parts shape: %+v", body)
	}
	inline := body.Contents[0].Parts[0].InlineData
	if inline == nil {
		t.Fatalf("inlineData missing")
	}
	if inline.MIMEType != "audio/wav" {
		t.Errorf("mimeType = %q", inline.MIMEType)
	}
	dec, err := base64.StdEncoding.DecodeString(inline.Data)
	if err != nil {
		t.Fatalf("base64: %v", err)
	}
	if !bytes.Equal(dec, audio) {
		t.Errorf("decoded audio mismatch: %q vs %q", dec, audio)
	}
}

func TestTranscribe_PromptIsAppended(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("ok")
	p := r.newProvider(t)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model:  "m",
		Audio:  bytes.NewReader([]byte("a")),
		Prompt: "Names: Alice, Bob.",
	})
	if !bytes.Contains(r.body, []byte("Names: Alice, Bob.")) {
		t.Errorf("body missing prompt: %s", string(r.body))
	}
	if !bytes.Contains(r.body, []byte("Transcribe this audio.")) {
		t.Errorf("body missing transcribe instruction")
	}
}

func TestTranscribe_LanguageInPrompt(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("ok")
	p := r.newProvider(t)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model:    "m",
		Audio:    bytes.NewReader([]byte("a")),
		Language: "fr",
	})
	if !bytes.Contains(r.body, []byte("Transcribe this fr audio.")) {
		t.Errorf("body missing language instruction: %s", string(r.body))
	}
}

func TestTranscribe_DefaultMimeWhenFormatEmpty(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("ok")
	p := r.newProvider(t)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "m",
		Audio: bytes.NewReader([]byte("a")),
	})
	if !bytes.Contains(r.body, []byte(`"mimeType":"audio/mpeg"`)) {
		t.Errorf("default mimeType missing: %s", string(r.body))
	}
}

func TestTranscribe_UpstreamError(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respCode = http.StatusBadRequest
	r.respBody = `{"error":"nope"}`
	p := r.newProvider(t)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "m",
		Audio: bytes.NewReader([]byte("a")),
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if up.StatusCode != 400 {
		t.Errorf("StatusCode = %d", up.StatusCode)
	}
}

func TestTranscribe_RequiresAudio(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{Model: "m"})
	if err == nil {
		t.Fatal("expected error on nil audio")
	}
}

func TestTranscribe_RequiresModel(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Audio: bytes.NewReader([]byte("a")),
	})
	if err == nil {
		t.Fatal("expected error on empty model")
	}
}

func TestTranscribe_NoCandidates(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = `{"candidates":[]}`
	p := r.newProvider(t)
	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "m",
		Audio: bytes.NewReader([]byte("a")),
	})
	if err == nil {
		t.Fatal("expected error on empty candidates")
	}
}

func TestTranscribe_HitsGenerateContentPath(t *testing.T) {
	r := newAudioRecorder(t)
	defer r.close()
	r.respBody = makeTranscribeResponse("ok")
	p := r.newProvider(t)
	_, _ = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "m",
		Audio: bytes.NewReader([]byte("a")),
	})
	if !strings.HasSuffix(r.urlPath, ":generateContent") {
		t.Errorf("path = %q, want :generateContent suffix", r.urlPath)
	}
}
