package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// generateContentSuffix is the URL suffix for the unary generateContent
// endpoint used for both TTS (via responseModalities=AUDIO) and STT (via
// inline audio + prompt) on Gemini AI Studio.
const generateContentSuffix = ":generateContent"

// Speak implements llmrouter.Speaker against Gemini AI Studio's
// generateContent endpoint with responseModalities=["AUDIO"]. Gemini TTS
// does not stream as of 2025; req.Stream is accepted but ignored — the
// entire audio arrives in a single AudioChunk.
//
// Gemini returns raw 16-bit signed little-endian PCM at 24 kHz
// (mimeType "audio/L16;rate=24000"). When req.Format == "wav", the bytes
// are wrapped in a RIFF/WAV header before delivery and ContentType is set
// to "audio/wav". Otherwise the bytes pass through with their reported
// MIME type.
func (p *Provider) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("gemini: speak requires model")
	}
	if req.Input == "" {
		return nil, fmt.Errorf("gemini: speak requires input")
	}

	body, err := buildSpeakBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build speak body: %w", err)
	}
	url := fmt.Sprintf("%s/models/%s%s", p.cfg.BaseURL, req.Model, generateContentSuffix)
	raw, err := p.doAudio(ctx, url, body)
	if err != nil {
		return nil, err
	}
	mime, audio, err := decodeSpeakResponse(raw)
	if err != nil {
		return nil, err
	}

	contentType, data := finalizeAudio(req.Format, mime, audio)

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = contentType
	go func() {
		if sctx.Err() != nil {
			hooks.Finish(sctx.Err())
			return
		}
		hooks.Send(llmrouter.AudioChunk{Data: data, Raw: raw})
		hooks.Finish(nil)
	}()
	return stream, nil
}

// buildSpeakBody assembles the generateContent JSON body for a TTS call.
// The Raw field is overlaid for vendor extras.
func buildSpeakBody(req llmrouter.SpeechRequest) ([]byte, error) {
	speechCfg := map[string]any{}
	if req.Voice != "" {
		speechCfg["voiceConfig"] = map[string]any{
			"prebuiltVoiceConfig": map[string]any{"voiceName": req.Voice},
		}
	}
	genCfg := map[string]any{
		"responseModalities": []string{"AUDIO"},
	}
	if len(speechCfg) > 0 {
		genCfg["speechConfig"] = speechCfg
	}
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]any{{"text": req.Input}}},
		},
		"generationConfig": genCfg,
	}
	return marshalSpeechWithRaw(body, req.Raw)
}

// marshalSpeechWithRaw overlays Raw extras onto body without clobbering
// known keys.
func marshalSpeechWithRaw(body map[string]any, raw json.RawMessage) ([]byte, error) {
	if len(raw) > 0 {
		var extra map[string]json.RawMessage
		if err := json.Unmarshal(raw, &extra); err != nil {
			return nil, fmt.Errorf("invalid raw body: %w", err)
		}
		for k, v := range extra {
			if _, exists := body[k]; exists {
				continue
			}
			body[k] = v
		}
	}
	return json.Marshal(body)
}

// generateContentSpeakWire mirrors the audio-bearing parts of
// generateContent's response.
type generateContentSpeakWire struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				InlineData *struct {
					MIMEType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData,omitempty"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// decodeSpeakResponse extracts (mimeType, audioBytes) from the first
// inlineData part across all candidates.
func decodeSpeakResponse(raw []byte) (string, []byte, error) {
	var w generateContentSpeakWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return "", nil, fmt.Errorf("gemini: decode speak response: %w", err)
	}
	for _, cand := range w.Candidates {
		for _, part := range cand.Content.Parts {
			if part.InlineData == nil || part.InlineData.Data == "" {
				continue
			}
			data, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
			if err != nil {
				return "", nil, fmt.Errorf("gemini: decode audio base64: %w", err)
			}
			return part.InlineData.MIMEType, data, nil
		}
	}
	return "", nil, fmt.Errorf("gemini: no audio in response")
}

// finalizeAudio applies WAV-wrapping when the caller asked for wav, and
// resolves the ContentType.
func finalizeAudio(format, sourceMIME string, data []byte) (string, []byte) {
	if strings.EqualFold(format, "wav") {
		rate := sampleRateFromMIME(sourceMIME)
		return "audio/wav", wrapPCMAsWAV(data, rate, 1, 16)
	}
	ct := sourceMIME
	if ct == "" {
		ct = "application/octet-stream"
	}
	return ct, data
}

// sampleRateFromMIME parses the "rate=<n>" parameter from a Gemini audio
// MIME type (e.g. "audio/L16;rate=24000"). Defaults to 24000.
func sampleRateFromMIME(mime string) int {
	const defaultRate = 24000
	for _, part := range strings.Split(mime, ";") {
		part = strings.TrimSpace(part)
		if !strings.HasPrefix(strings.ToLower(part), "rate=") {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(part[len("rate="):], "%d", &n); err == nil && n > 0 {
			return n
		}
	}
	return defaultRate
}

// wrapPCMAsWAV prepends a minimal canonical RIFF/WAVE header to a raw PCM
// payload. The audio is assumed to be 16-bit signed little-endian.
func wrapPCMAsWAV(pcm []byte, sampleRate, channels, bitsPerSample int) []byte {
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := len(pcm)

	buf := bytes.NewBuffer(make([]byte, 0, 44+dataSize))
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))           // PCM fmt chunk size
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))            // audio format = PCM
	_ = binary.Write(buf, binary.LittleEndian, uint16(channels))     //
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))   //
	_ = binary.Write(buf, binary.LittleEndian, uint32(byteRate))     //
	_ = binary.Write(buf, binary.LittleEndian, uint16(blockAlign))   //
	_ = binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))//
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	buf.Write(pcm)
	return buf.Bytes()
}

// Transcribe implements llmrouter.Transcriber against Gemini AI Studio's
// generateContent endpoint with an inline audio part + transcription
// prompt. For v0.3 this uses inline base64 audio only — Gemini's 20 MiB
// inline limit applies; for larger inputs the Files API would be needed.
//
// The response is emitted as a single Final TranscriptSegment containing
// the model's text. Gemini does not return word-level timing in this mode.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("gemini: transcribe requires model")
	}
	if req.Audio == nil {
		return nil, fmt.Errorf("gemini: transcribe requires audio")
	}

	audioBytes, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("gemini: read audio: %w", err)
	}

	body, err := buildTranscribeBody(req, audioBytes)
	if err != nil {
		return nil, fmt.Errorf("gemini: build transcribe body: %w", err)
	}
	url := fmt.Sprintf("%s/models/%s%s", p.cfg.BaseURL, req.Model, generateContentSuffix)
	raw, err := p.doAudio(ctx, url, body)
	if err != nil {
		return nil, err
	}
	text, err := decodeTranscribeResponse(raw)
	if err != nil {
		return nil, err
	}

	stream, sctx, hooks := llmrouter.NewTranscriptStream(ctx)
	go func() {
		if sctx.Err() != nil {
			hooks.Finish(sctx.Err())
			return
		}
		hooks.Send(llmrouter.TranscriptSegment{
			Text:  text,
			Final: true,
			Raw:   json.RawMessage(raw),
		})
		hooks.Finish(nil)
	}()
	return stream, nil
}

// buildTranscribeBody composes the inline-audio + text prompt request body.
func buildTranscribeBody(req llmrouter.TranscribeRequest, audio []byte) ([]byte, error) {
	mime := req.AudioFormat
	if mime == "" {
		mime = "audio/mpeg"
	}
	prompt := transcribePrompt(req)
	parts := []map[string]any{
		{"inlineData": map[string]any{
			"mimeType": mime,
			"data":     base64.StdEncoding.EncodeToString(audio),
		}},
		{"text": prompt},
	}
	body := map[string]any{
		"contents": []map[string]any{{"parts": parts}},
	}
	return marshalSpeechWithRaw(body, req.Raw)
}

// transcribePrompt builds the instruction text, optionally mentioning the
// declared language and appending the caller's prompt for context.
func transcribePrompt(req llmrouter.TranscribeRequest) string {
	var base string
	if req.Language != "" {
		base = "Transcribe this " + req.Language + " audio."
	} else {
		base = "Transcribe this audio."
	}
	if req.Prompt != "" {
		base = base + " " + req.Prompt
	}
	return base
}

// generateContentTextWire mirrors the text-bearing parts of
// generateContent's response.
type generateContentTextWire struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text,omitempty"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// decodeTranscribeResponse collects all text parts across the first
// candidate into one transcript string.
func decodeTranscribeResponse(raw []byte) (string, error) {
	var w generateContentTextWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return "", fmt.Errorf("gemini: decode transcribe response: %w", err)
	}
	if len(w.Candidates) == 0 {
		return "", fmt.Errorf("gemini: no candidates in transcribe response")
	}
	var sb strings.Builder
	for _, part := range w.Candidates[0].Content.Parts {
		sb.WriteString(part.Text)
	}
	return sb.String(), nil
}

// doAudio performs the POST for audio-shaped generateContent calls and
// maps non-2xx responses to *llmrouter.ErrUpstream.
func (p *Provider) doAudio(ctx context.Context, url string, body []byte) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set(apiKeyHeader, p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("gemini: http: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyCap))
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       string(b),
		}
	}
	return io.ReadAll(resp.Body)
}
