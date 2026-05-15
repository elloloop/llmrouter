package vertex

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"

	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// audioWAVDefaultRate is the fallback sample rate when the upstream MIME
// type omits a rate parameter. Vertex Gemini TTS uses 24 kHz.
const audioWAVDefaultRate = 24000

// Speak implements llmrouter.Speaker against Vertex's GenerateContent
// method with responseModalities=["AUDIO"]. The call is non-streaming —
// the entire audio is returned as one AudioChunk regardless of
// req.Stream.
//
// When req.Format == "wav", the raw PCM payload is wrapped in a RIFF/WAVE
// header before delivery and ContentType becomes "audio/wav". Otherwise
// the bytes pass through with the upstream MIME type.
func (p *Provider) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("vertex: speak requires model")
	}
	if req.Input == "" {
		return nil, fmt.Errorf("vertex: speak requires input")
	}

	contents := []*genai.Content{{
		Role:  roleUser,
		Parts: []*genai.Part{{Text: req.Input}},
	}}
	cfg := buildSpeakConfig(req)

	resp, err := p.client.Models.GenerateContent(ctx, req.Model, contents, cfg)
	if err != nil {
		return nil, wrapSDKError(err)
	}
	mime, audio, err := extractFirstInlineAudio(resp)
	if err != nil {
		return nil, fmt.Errorf("vertex: %w", err)
	}

	contentType, data := finalizeAudio(req.Format, mime, audio)

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = contentType
	go func() {
		if sctx.Err() != nil {
			hooks.Finish(sctx.Err())
			return
		}
		hooks.Send(llmrouter.AudioChunk{Data: data, Raw: data})
		hooks.Finish(nil)
	}()
	return stream, nil
}

// buildSpeakConfig assembles a GenerateContentConfig for TTS, including
// the AUDIO response modality and an optional prebuilt voice.
func buildSpeakConfig(req llmrouter.SpeechRequest) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{
		ResponseModalities: []string{"AUDIO"},
	}
	if req.Voice != "" {
		cfg.SpeechConfig = &genai.SpeechConfig{
			VoiceConfig: &genai.VoiceConfig{
				PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: req.Voice},
			},
		}
	}
	return cfg
}

// extractFirstInlineAudio walks the response candidates and returns the
// first part carrying inline audio bytes plus its MIME type.
func extractFirstInlineAudio(resp *genai.GenerateContentResponse) (string, []byte, error) {
	if resp == nil {
		return "", nil, errors.New("nil response")
	}
	for _, cand := range resp.Candidates {
		if cand == nil || cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part == nil || part.InlineData == nil {
				continue
			}
			if len(part.InlineData.Data) == 0 {
				continue
			}
			return part.InlineData.MIMEType, part.InlineData.Data, nil
		}
	}
	return "", nil, errors.New("no inline audio in response")
}

// finalizeAudio applies WAV wrapping when requested and resolves the
// outgoing ContentType.
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

// sampleRateFromMIME extracts the "rate=<n>" parameter from a Gemini audio
// MIME type (e.g. "audio/L16;rate=24000"). Defaults to audioWAVDefaultRate.
func sampleRateFromMIME(mime string) int {
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
	return audioWAVDefaultRate
}

// wrapPCMAsWAV prepends a minimal canonical RIFF/WAVE header to a raw PCM
// payload. PCM is assumed to be 16-bit signed little-endian.
func wrapPCMAsWAV(pcm []byte, sampleRate, channels, bitsPerSample int) []byte {
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := len(pcm)

	buf := bytes.NewBuffer(make([]byte, 0, 44+dataSize))
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1)) // PCM
	_ = binary.Write(buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(buf, binary.LittleEndian, uint32(byteRate))
	_ = binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	_ = binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	buf.Write(pcm)
	return buf.Bytes()
}

// Transcribe implements llmrouter.Transcriber against Vertex's
// GenerateContent endpoint with an inline audio part + transcription
// prompt. Inline audio only — for >20 MiB inputs, use the Files API
// directly. The response is emitted as one Final TranscriptSegment.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("vertex: transcribe requires model")
	}
	if req.Audio == nil {
		return nil, fmt.Errorf("vertex: transcribe requires audio")
	}

	audioBytes, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("vertex: read audio: %w", err)
	}

	mime := req.AudioFormat
	if mime == "" {
		mime = "audio/mpeg"
	}
	prompt := transcribePrompt(req)
	contents := []*genai.Content{{
		Role: roleUser,
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{Data: audioBytes, MIMEType: mime}},
			{Text: prompt},
		},
	}}

	resp, err := p.client.Models.GenerateContent(ctx, req.Model, contents, nil)
	if err != nil {
		return nil, wrapSDKError(err)
	}
	text, err := extractTranscriptText(resp)
	if err != nil {
		return nil, fmt.Errorf("vertex: %w", err)
	}

	stream, sctx, hooks := llmrouter.NewTranscriptStream(ctx)
	go func() {
		if sctx.Err() != nil {
			hooks.Finish(sctx.Err())
			return
		}
		hooks.Send(llmrouter.TranscriptSegment{Text: text, Final: true})
		hooks.Finish(nil)
	}()
	return stream, nil
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

// extractTranscriptText concatenates all text parts of the first candidate.
func extractTranscriptText(resp *genai.GenerateContentResponse) (string, error) {
	if resp == nil || len(resp.Candidates) == 0 {
		return "", errors.New("no candidates in response")
	}
	cand := resp.Candidates[0]
	if cand == nil || cand.Content == nil {
		return "", errors.New("empty candidate content")
	}
	var sb strings.Builder
	for _, part := range cand.Content.Parts {
		if part == nil {
			continue
		}
		sb.WriteString(part.Text)
	}
	return sb.String(), nil
}
