package llmrouter

import (
	"context"
	"encoding/json"
	"io"
	"time"
)

// Speaker generates audio from text (TTS). Implementations are
// concurrency-safe.
type Speaker interface {
	Speak(ctx context.Context, req SpeechRequest) (*AudioStream, error)
}

// Transcriber turns audio bytes into text (STT). Implementations are
// concurrency-safe.
type Transcriber interface {
	Transcribe(ctx context.Context, req TranscribeRequest) (*TranscriptStream, error)
}

// SpeechRequest describes a TTS call. Format defaults to "mp3" when empty.
type SpeechRequest struct {
	// Model identifier (e.g. "tts-1", "tts-1-hd", "eleven_turbo_v2_5").
	Model string `json:"model"`

	// Input text to synthesise. Required.
	Input string `json:"input"`

	// Voice identifier (e.g. "alloy", "echo", or a vendor-specific voice id).
	Voice string `json:"voice,omitempty"`

	// Format is the desired audio container/codec:
	//   "mp3" (default) | "opus" | "aac" | "flac" | "wav" | "pcm" | "ulaw"
	// Providers map this to their native enum.
	Format string `json:"response_format,omitempty"`

	// Speed is the playback-rate multiplier (1.0 = normal). Range 0.25-4.0.
	Speed *float64 `json:"speed,omitempty"`

	// Stream requests chunked audio. When true, AudioStream may emit many
	// chunks; when false, the entire audio arrives as a single chunk.
	Stream bool `json:"stream,omitempty"`

	// Raw is forwarded as the outgoing JSON body for providers that
	// implement byte passthrough. The library overlays Model and Input.
	Raw json.RawMessage `json:"-"`
}

// AudioStream delivers TTS audio bytes in chunks. The stream closes when
// the upstream finishes, the context cancels, or an error occurs.
// Single-consumer; one goroutine reads Chunks().
type AudioStream struct {
	chunks chan AudioChunk
	cancel context.CancelFunc
	errMu  chan struct{}
	err    error
	// ContentType is the audio MIME type ("audio/mpeg", "audio/opus",
	// "audio/wav", "audio/pcm", etc.). Populated by the producer before
	// any chunks are sent.
	ContentType string
}

// AudioChunk is one frame of audio bytes from an AudioStream.
type AudioChunk struct {
	// Data is the audio bytes. For non-streaming providers the entire
	// audio arrives as a single chunk.
	Data []byte
	// Raw is the original wire bytes; for many providers Data == Raw.
	Raw []byte
}

// Chunks returns the receive-only chunk channel. Same single-consumer
// contract as llmrouter.Stream.
func (s *AudioStream) Chunks() <-chan AudioChunk { return s.chunks }

// Err blocks until the producer finishes and returns the terminal error
// (nil on success). Safe to call multiple times; returns the same value.
func (s *AudioStream) Err() error {
	<-s.errMu
	return s.err
}

// Cancel asks the producer to stop. Idempotent.
func (s *AudioStream) Cancel() {
	if s.cancel != nil {
		s.cancel()
	}
}

// NewAudioStream is the provider-facing constructor — mirrors NewStream.
// Returns the stream, a derived context the producer should respect, and
// two callback hooks the producer must use: Send and Finish (Finish exactly
// once). If the producer needs to set ContentType, it must do so on the
// returned *AudioStream BEFORE calling Send.
func NewAudioStream(parent context.Context) (*AudioStream, context.Context, AudioProducerHooks) {
	ctx, cancel := context.WithCancel(parent)
	s := &AudioStream{
		chunks: make(chan AudioChunk, 16),
		cancel: cancel,
		errMu:  make(chan struct{}),
	}
	sendFn := func(c AudioChunk) bool {
		select {
		case s.chunks <- c:
			return true
		case <-ctx.Done():
			return false
		}
	}
	finishFn := func(err error) {
		s.err = err
		close(s.chunks)
		close(s.errMu)
	}
	return s, ctx, AudioProducerHooks{Send: sendFn, Finish: finishFn}
}

// AudioProducerHooks are the callbacks a Speaker implementation uses to
// feed an AudioStream.
type AudioProducerHooks struct {
	Send   func(AudioChunk) bool
	Finish func(error)
}

// TranscribeRequest describes an STT call.
type TranscribeRequest struct {
	// Model identifier (e.g. "whisper-1", "nova-2", "scribe_v1").
	Model string `json:"model"`

	// Audio is the binary audio source. Required. The library reads until
	// EOF; rewinding is the caller's responsibility for retries.
	Audio io.Reader `json:"-"`

	// AudioFormat is the source MIME type ("audio/mpeg", "audio/wav",
	// "audio/webm", "audio/m4a", "audio/flac", "audio/ogg", ...). Some
	// providers ignore this and sniff; others require it.
	AudioFormat string `json:"-"`

	// Filename is optional metadata used by multipart-upload providers
	// (OpenAI, Azure). Defaults to "audio" + an extension derived from
	// AudioFormat when empty.
	Filename string `json:"-"`

	// Language is an optional ISO-639-1 hint ("en", "fr", ...). Speeds up
	// detection and improves accuracy.
	Language string `json:"language,omitempty"`

	// Prompt is an optional context hint (OpenAI Whisper, ElevenLabs).
	Prompt string `json:"prompt,omitempty"`

	// ResponseFormat — "json" (default) | "text" | "srt" | "vtt" |
	// "verbose_json". The library normalises to TranscriptSegment[] either
	// way, but providers may produce richer detail in verbose modes.
	ResponseFormat string `json:"response_format,omitempty"`

	// Temperature is the sampling temperature (Whisper supports this).
	Temperature *float64 `json:"temperature,omitempty"`

	// Stream requests live transcription. Providers that don't support
	// streaming send a single final segment regardless.
	Stream bool `json:"stream,omitempty"`

	// Raw is forwarded for passthrough callers (overlaid by Model).
	Raw json.RawMessage `json:"-"`
}

// TranscriptStream emits transcript text in one or more segments. For
// non-streaming providers, exactly one segment arrives with Final=true.
// Single-consumer; one goroutine reads Segments().
type TranscriptStream struct {
	segments chan TranscriptSegment
	cancel   context.CancelFunc
	errMu    chan struct{}
	err      error
}

// TranscriptSegment is one piece of transcribed text. Streaming providers
// emit interim segments (Final=false) followed by a final segment
// (Final=true). Non-streaming providers emit one Final segment.
type TranscriptSegment struct {
	// Text is the transcribed text for this segment.
	Text string `json:"text"`

	// Final is true on the terminal segment.
	Final bool `json:"final"`

	// Start / End are timestamps relative to the start of the audio.
	// Zero when the provider doesn't return timing.
	Start time.Duration `json:"start,omitempty"`
	End   time.Duration `json:"end,omitempty"`

	// Words optionally carries per-word timing (Whisper verbose_json,
	// Deepgram, ElevenLabs). May be empty.
	Words []TranscriptWord `json:"words,omitempty"`

	// Confidence is the provider's per-segment confidence in [0,1].
	// Zero when not provided.
	Confidence float32 `json:"confidence,omitempty"`

	// Raw is the original wire-format JSON for this segment.
	Raw json.RawMessage `json:"-"`
}

// TranscriptWord is one word with timing.
type TranscriptWord struct {
	Word       string        `json:"word"`
	Start      time.Duration `json:"start,omitempty"`
	End        time.Duration `json:"end,omitempty"`
	Confidence float32       `json:"confidence,omitempty"`
}

// Segments returns the receive-only segment channel.
func (s *TranscriptStream) Segments() <-chan TranscriptSegment { return s.segments }

// Err blocks until the producer finishes; safe to call multiple times.
func (s *TranscriptStream) Err() error {
	<-s.errMu
	return s.err
}

// Cancel asks the producer to stop. Idempotent.
func (s *TranscriptStream) Cancel() {
	if s.cancel != nil {
		s.cancel()
	}
}

// NewTranscriptStream is the provider-facing constructor.
func NewTranscriptStream(parent context.Context) (*TranscriptStream, context.Context, TranscriptProducerHooks) {
	ctx, cancel := context.WithCancel(parent)
	s := &TranscriptStream{
		segments: make(chan TranscriptSegment, 16),
		cancel:   cancel,
		errMu:    make(chan struct{}),
	}
	sendFn := func(seg TranscriptSegment) bool {
		select {
		case s.segments <- seg:
			return true
		case <-ctx.Done():
			return false
		}
	}
	finishFn := func(err error) {
		s.err = err
		close(s.segments)
		close(s.errMu)
	}
	return s, ctx, TranscriptProducerHooks{Send: sendFn, Finish: finishFn}
}

// TranscriptProducerHooks are the callbacks a Transcriber implementation
// uses to feed a TranscriptStream.
type TranscriptProducerHooks struct {
	Send   func(TranscriptSegment) bool
	Finish func(error)
}
