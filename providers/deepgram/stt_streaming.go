package deepgram

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
)

const (
	// liveAudioChunkBytes bounds each binary frame written to the
	// Deepgram websocket. Deepgram recommends keeping individual audio
	// frames below 8 KiB; we use 4 KiB to stay comfortably inside.
	liveAudioChunkBytes = 4 * 1024

	// defaultEndpointingMs is the silence threshold (in milliseconds)
	// Deepgram uses to decide when an utterance has ended.
	defaultEndpointingMs = "300"
)

// rawLiveQueryKeys is the additional allow-list of query parameters that
// only make sense on the live (websocket) endpoint. Batch keys from
// rawQueryKeys are also forwarded.
var rawLiveQueryKeys = []string{
	"interim_results",
	"endpointing",
	"vad_events",
	"utterance_end_ms",
	"no_delay",
	"encoding",
	"sample_rate",
	"channels",
}

// transcribeStreaming opens a websocket to Deepgram's live /v1/listen
// endpoint, streams the request audio in real-time-paced binary frames,
// and emits TranscriptSegments as Results frames arrive.
func (p *Provider) transcribeStreaming(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	wsURL, err := buildLiveURL(p.cfg.BaseURL, req)
	if err != nil {
		return nil, err
	}

	headers := http.Header{}
	headers.Set("Authorization", "Token "+p.cfg.APIKey)

	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: headers,
	})
	if err != nil {
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       "deepgram dial: " + err.Error(),
		}
	}
	// Allow arbitrarily large transcript frames; defaults are tight
	// enough that long utterances can otherwise blow the limit.
	conn.SetReadLimit(-1)

	stream, sctx, hooks := llmrouter.NewTranscriptStream(ctx)

	// done coordinates the two pumps so the receiver can call Finish
	// exactly once, after the sender has stopped writing. We use a
	// dedicated channel rather than a sync.Once because we also want the
	// receiver to wait for the sender's CloseStream sentinel before
	// declaring the conversation over.
	go pumpDeepgramSender(sctx, conn, req.Audio)
	go pumpDeepgramReceiver(sctx, conn, hooks)

	return stream, nil
}

// pumpDeepgramReceiver reads JSON text frames off the websocket, decodes
// each Results event into a TranscriptSegment, and pushes it through the
// supplied hooks. It always calls hooks.Finish exactly once and always
// closes the connection on exit.
func pumpDeepgramReceiver(ctx context.Context, conn *websocket.Conn, hooks llmrouter.TranscriptProducerHooks) {
	var finishErr error
	defer func() {
		hooks.Finish(finishErr)
		// Best-effort normal close; ignore any close error since the
		// terminal state is already reflected in finishErr.
		_ = conn.Close(websocket.StatusNormalClosure, "client done")
	}()

	for {
		msgType, payload, err := conn.Read(ctx)
		if err != nil {
			if isExpectedCloseErr(err) {
				return
			}
			finishErr = &llmrouter.ErrUpstream{
				Provider:   providerName,
				StatusCode: 0,
				Body:       "deepgram ws read: " + err.Error(),
			}
			return
		}
		if msgType != websocket.MessageText {
			// Deepgram only ever sends text frames; defensively skip
			// any binary frames the server might emit.
			continue
		}
		seg, ok, err := decodeLiveFrame(payload)
		if err != nil {
			finishErr = err
			return
		}
		if !ok {
			continue
		}
		if !hooks.Send(seg) {
			// Consumer cancelled — stop reading.
			return
		}
	}
}

// pumpDeepgramSender copies audio bytes from src to the websocket as
// binary frames of at most liveAudioChunkBytes each. On EOF it sends the
// CloseStream sentinel so Deepgram flushes a final transcript before
// closing. On context cancellation it just closes the connection.
func pumpDeepgramSender(ctx context.Context, conn *websocket.Conn, src io.Reader) {
	if src == nil {
		_ = sendCloseStream(ctx, conn)
		return
	}
	reader := bufio.NewReader(src)
	buf := make([]byte, liveAudioChunkBytes)
	for {
		if ctx.Err() != nil {
			_ = conn.Close(websocket.StatusNormalClosure, "context cancelled")
			return
		}
		n, err := reader.Read(buf)
		if n > 0 {
			if writeErr := conn.Write(ctx, websocket.MessageBinary, buf[:n]); writeErr != nil {
				return
			}
		}
		if err == nil {
			continue
		}
		if errors.Is(err, io.EOF) {
			_ = sendCloseStream(ctx, conn)
			return
		}
		// Unknown read error from the audio source — abort the stream.
		_ = conn.Close(websocket.StatusInternalError, "audio read error")
		return
	}
}

// sendCloseStream writes the JSON sentinel that tells Deepgram no more
// audio is coming so the server can flush the final transcript before
// closing. Errors are returned for the caller to ignore — the underlying
// connection will be closed by the receiver.
func sendCloseStream(ctx context.Context, conn *websocket.Conn) error {
	return conn.Write(ctx, websocket.MessageText, []byte(`{"type":"CloseStream"}`))
}

// isExpectedCloseErr reports whether err represents one of the close
// signals we treat as a clean end-of-stream.
func isExpectedCloseErr(err error) bool {
	if errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) {
		return true
	}
	closeStatus := websocket.CloseStatus(err)
	switch closeStatus {
	case websocket.StatusNormalClosure, websocket.StatusGoingAway, websocket.StatusNoStatusRcvd:
		return true
	}
	return false
}

// buildLiveURL produces a wss:// URL pointing at /v1/listen with all the
// query parameters Deepgram needs to start a live transcription session.
func buildLiveURL(baseURL string, req llmrouter.TranscribeRequest) (string, error) {
	u, err := url.Parse(baseURL + "/v1/listen")
	if err != nil {
		return "", fmt.Errorf("deepgram: parse base url: %w", err)
	}
	switch strings.ToLower(u.Scheme) {
	case "https":
		u.Scheme = "wss"
	case "http":
		u.Scheme = "ws"
	}

	model := req.Model
	if model == "" {
		model = defaultModel
	}
	language := req.Language
	if language == "" {
		language = defaultLanguage
	}

	q := u.Query()
	q.Set("model", model)
	q.Set("language", language)
	q.Set("punctuate", "true")
	q.Set("smart_format", "true")
	q.Set("utterances", "true")
	q.Set("interim_results", "true")
	q.Set("endpointing", defaultEndpointingMs)

	if encoding, sampleRate, ok := encodingFor(req.AudioFormat); ok {
		q.Set("encoding", encoding)
		if sampleRate != "" {
			q.Set("sample_rate", sampleRate)
		}
		q.Set("channels", "1")
	}

	mergeRawQueryKeys(q, req.Raw, rawQueryKeys)
	mergeRawQueryKeys(q, req.Raw, rawLiveQueryKeys)

	u.RawQuery = q.Encode()
	return u.String(), nil
}

// encodingFor maps a few common AudioFormat MIME types to Deepgram's
// (encoding, sample_rate) live-API parameters. Returns ok=false for
// formats Deepgram detects from the container itself (opus container,
// webm, ogg, mp3) so we leave encoding unset.
func encodingFor(audioFormat string) (encoding, sampleRate string, ok bool) {
	switch strings.ToLower(audioFormat) {
	case "audio/wav", "audio/x-wav", "audio/wave", "audio/l16":
		return "linear16", "16000", true
	case "audio/pcm":
		return "linear16", "16000", true
	case "audio/mulaw", "audio/x-mulaw", "audio/basic":
		return "mulaw", "8000", true
	case "audio/alaw", "audio/x-alaw":
		return "alaw", "8000", true
	case "audio/flac":
		return "flac", "", true
	case "audio/opus":
		return "opus", "", true
	case "audio/amr":
		return "amr-nb", "8000", true
	}
	return "", "", false
}

// mergeRawQueryKeys is the streaming-aware analogue of mergeRawQuery
// from the batch path: it accepts an explicit allow-list so we can
// share the helper between the batch keys (rawQueryKeys) and the
// live-only keys (rawLiveQueryKeys).
func mergeRawQueryKeys(q url.Values, raw json.RawMessage, keys []string) {
	if len(raw) == 0 {
		return
	}
	var src map[string]json.RawMessage
	if err := json.Unmarshal(raw, &src); err != nil {
		return
	}
	for _, key := range keys {
		v, ok := src[key]
		if !ok {
			continue
		}
		s, ok := scalarToString(v)
		if !ok {
			continue
		}
		q.Set(key, s)
	}
}

// deepgramLiveFrame is the subset of a live Results event the provider
// consumes. Unknown fields are ignored.
type deepgramLiveFrame struct {
	Type         string  `json:"type"`
	Start        float64 `json:"start"`
	Duration     float64 `json:"duration"`
	IsFinal      bool    `json:"is_final"`
	SpeechFinal  bool    `json:"speech_final"`
	ChannelIndex []int   `json:"channel_index"`
	Channel      struct {
		Alternatives []struct {
			Transcript string         `json:"transcript"`
			Confidence float64        `json:"confidence"`
			Words      []deepgramWord `json:"words"`
		} `json:"alternatives"`
	} `json:"channel"`
}

// decodeLiveFrame converts one wire payload into a TranscriptSegment.
// Returns ok=false for events the caller should silently skip
// (Metadata, SpeechStarted, UtteranceEnd, anything without a
// transcript). Returns a non-nil error only when the JSON itself is
// malformed.
func decodeLiveFrame(payload []byte) (llmrouter.TranscriptSegment, bool, error) {
	var frame deepgramLiveFrame
	if err := json.Unmarshal(payload, &frame); err != nil {
		return llmrouter.TranscriptSegment{}, false, fmt.Errorf("deepgram: decode live frame: %w", err)
	}
	if frame.Type != "Results" {
		return llmrouter.TranscriptSegment{}, false, nil
	}
	if len(frame.Channel.Alternatives) == 0 {
		return llmrouter.TranscriptSegment{}, false, nil
	}
	alt := frame.Channel.Alternatives[0]
	rawCopy := make(json.RawMessage, len(payload))
	copy(rawCopy, payload)
	seg := llmrouter.TranscriptSegment{
		Text:       alt.Transcript,
		Final:      frame.IsFinal,
		Start:      secondsToDuration(frame.Start),
		End:        secondsToDuration(frame.Start + frame.Duration),
		Confidence: float32(alt.Confidence),
		Words:      mapWords(alt.Words),
		Raw:        rawCopy,
	}
	return seg, true, nil
}

