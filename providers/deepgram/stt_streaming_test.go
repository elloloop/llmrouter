package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
)

// liveServerOpts configures the fake Deepgram live server used by the
// streaming tests. Each test gets its own server with its own script.
type liveServerOpts struct {
	// rejectHandshake, when non-zero, causes the server to respond with
	// this HTTP status during the websocket upgrade.
	rejectHandshake int

	// pauseBeforeFinal makes the server delay the final Results frame
	// by the given duration; useful for context-cancellation tests.
	pauseBeforeFinal time.Duration

	// holdOpen keeps the connection open after sending the script (no
	// close, no further frames) so tests can exercise client-driven
	// cancellation.
	holdOpen bool
}

// liveServerCapture records everything the fake server observed during a
// single websocket session.
type liveServerCapture struct {
	mu             sync.Mutex
	upgradeHeaders http.Header
	upgradeURL     string
	binaryFrames   [][]byte
	textFrames     []string
	closeReceived  bool
	connectCount   int
}

func (c *liveServerCapture) appendBinary(b []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	cp := make([]byte, len(b))
	copy(cp, b)
	c.binaryFrames = append(c.binaryFrames, cp)
}

func (c *liveServerCapture) appendText(s string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.textFrames = append(c.textFrames, s)
	if strings.Contains(s, "CloseStream") {
		c.closeReceived = true
	}
}

// liveServerSnapshot is a lock-free copy of the captured state.
type liveServerSnapshot struct {
	upgradeHeaders http.Header
	upgradeURL     string
	binaryFrames   [][]byte
	textFrames     []string
	closeReceived  bool
	connectCount   int
}

func (c *liveServerCapture) snapshot() liveServerSnapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := liveServerSnapshot{
		upgradeHeaders: c.upgradeHeaders.Clone(),
		upgradeURL:     c.upgradeURL,
		closeReceived:  c.closeReceived,
		connectCount:   c.connectCount,
	}
	out.binaryFrames = append(out.binaryFrames, c.binaryFrames...)
	out.textFrames = append(out.textFrames, c.textFrames...)
	return out
}

// newLiveTestServer returns a fake Deepgram websocket server, a capture
// struct the test can read after the handler completes, and a Provider
// pointed at it. The server scripts a typical interim+final exchange:
// it reads inbound binary frames until it sees the CloseStream sentinel
// (or the client disconnects), then writes the configured Results
// frames. Tests can wait for completion using the returned done channel.
func newLiveTestServer(t *testing.T, opts liveServerOpts, resultsFrames []string) (*Provider, *liveServerCapture, <-chan struct{}) {
	t.Helper()
	cap := &liveServerCapture{}
	done := make(chan struct{}, 1)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.mu.Lock()
		cap.upgradeHeaders = r.Header.Clone()
		cap.upgradeURL = r.URL.String()
		cap.connectCount++
		cap.mu.Unlock()

		if opts.rejectHandshake != 0 {
			w.WriteHeader(opts.rejectHandshake)
			_, _ = w.Write([]byte(`{"error":"unauthorized"}`))
			select {
			case done <- struct{}{}:
			default:
			}
			return
		}

		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
		})
		if err != nil {
			t.Logf("accept: %v", err)
			select {
			case done <- struct{}{}:
			default:
			}
			return
		}
		conn.SetReadLimit(-1)

		ctx := r.Context()

		// Drain inbound frames in a goroutine. Signal closeSeen once
		// the client's CloseStream sentinel is received so the script
		// can safely emit a final transcript before closing.
		readDone := make(chan struct{})
		closeSeen := make(chan struct{})
		var closeOnce sync.Once
		go func() {
			defer close(readDone)
			for {
				typ, payload, err := conn.Read(ctx)
				if err != nil {
					closeOnce.Do(func() { close(closeSeen) })
					return
				}
				switch typ {
				case websocket.MessageBinary:
					cap.appendBinary(payload)
				case websocket.MessageText:
					cap.appendText(string(payload))
					if strings.Contains(string(payload), "CloseStream") {
						closeOnce.Do(func() { close(closeSeen) })
					}
				}
			}
		}()

		// Send the scripted frames. The final frame is held until the
		// client either sends CloseStream or disconnects, mirroring
		// Deepgram's real behaviour. pauseBeforeFinal additionally
		// inserts a fixed delay before the final frame for tests that
		// need a cancellation window.
		for i, frame := range resultsFrames {
			isLast := i == len(resultsFrames)-1
			if isLast {
				// Wait for client CloseStream before emitting the
				// final transcript, unless the test wants to keep
				// the connection open.
				if !opts.holdOpen {
					select {
					case <-closeSeen:
					case <-ctx.Done():
						select {
						case done <- struct{}{}:
						default:
						}
						return
					}
				}
				if opts.pauseBeforeFinal > 0 {
					select {
					case <-time.After(opts.pauseBeforeFinal):
					case <-ctx.Done():
						_ = conn.Close(websocket.StatusNormalClosure, "ctx done")
						select {
						case done <- struct{}{}:
						default:
						}
						return
					}
				}
			}
			if err := conn.Write(ctx, websocket.MessageText, []byte(frame)); err != nil {
				select {
				case done <- struct{}{}:
				default:
				}
				return
			}
		}

		if !opts.holdOpen {
			_ = conn.Close(websocket.StatusNormalClosure, "")
		}
		<-readDone
		select {
		case done <- struct{}{}:
		default:
		}
	})

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	provider, err := New(
		llmrouter.WithAPIKey("dg-test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("unexpected error building provider: %v", err)
	}
	return provider, cap, done
}

// resultFrame builds a Deepgram Results JSON payload for a transcript.
func resultFrame(transcript string, start, duration float64, isFinal bool, words []deepgramWord) string {
	type alt struct {
		Transcript string         `json:"transcript"`
		Confidence float64        `json:"confidence"`
		Words      []deepgramWord `json:"words"`
	}
	type channel struct {
		Alternatives []alt `json:"alternatives"`
	}
	out := map[string]any{
		"type":          "Results",
		"start":         start,
		"duration":      duration,
		"is_final":      isFinal,
		"speech_final":  isFinal,
		"channel_index": []int{0, 1},
		"channel": channel{
			Alternatives: []alt{{
				Transcript: transcript,
				Confidence: 0.97,
				Words:      words,
			}},
		},
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func TestTranscribeStreaming(t *testing.T) {
	frames := []string{
		resultFrame("hello", 0.0, 0.5, false, nil),
		resultFrame("hello world", 0.0, 1.0, false, nil),
		resultFrame("hello world.", 0.0, 1.04, true, []deepgramWord{
			{Word: "hello", Start: 0.0, End: 0.5, Confidence: 0.99, PunctuatedWord: "Hello"},
			{Word: "world", Start: 0.55, End: 1.0, Confidence: 0.97, PunctuatedWord: "world."},
		}),
	}

	t.Run("end_to_end_interim_and_final_segments", func(t *testing.T) {
		p, _, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("chunk1chunk2chunk3"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		<-done
		if len(segs) != 3 {
			t.Fatalf("expected 3 segments, got %d", len(segs))
		}
		if segs[0].Final || segs[1].Final {
			t.Fatalf("expected first two segments interim, got %+v", segs[:2])
		}
		if !segs[2].Final {
			t.Fatal("expected last segment Final=true")
		}
		if segs[2].Text != "hello world." {
			t.Fatalf("expected final text 'hello world.', got %q", segs[2].Text)
		}
		if len(segs[2].Words) != 2 {
			t.Fatalf("expected 2 words on final segment, got %d", len(segs[2].Words))
		}
	})

	t.Run("authorization_header_uses_token_not_bearer", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		got := snap.upgradeHeaders.Get("Authorization")
		if got != "Token dg-test-key" {
			t.Fatalf("expected Authorization=%q, got %q", "Token dg-test-key", got)
		}
		if strings.HasPrefix(got, "Bearer ") {
			t.Fatalf("Authorization must not use Bearer prefix, got %q", got)
		}
	})

	t.Run("url_contains_streaming_flags", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "interim_results=true") {
			t.Fatalf("expected interim_results=true in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "endpointing=300") {
			t.Fatalf("expected endpointing=300 in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "model=nova-3") {
			t.Fatalf("expected default model nova-3 in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "language=en") {
			t.Fatalf("expected default language en in URL, got %q", snap.upgradeURL)
		}
		for _, fixed := range []string{"punctuate=true", "smart_format=true", "utterances=true"} {
			if !strings.Contains(snap.upgradeURL, fixed) {
				t.Fatalf("expected %s in URL, got %q", fixed, snap.upgradeURL)
			}
		}
	})

	t.Run("wav_format_maps_to_linear16_16000", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:       strings.NewReader("a"),
			AudioFormat: "audio/wav",
			Stream:      true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "encoding=linear16") {
			t.Fatalf("expected encoding=linear16 in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "sample_rate=16000") {
			t.Fatalf("expected sample_rate=16000 in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "channels=1") {
			t.Fatalf("expected channels=1 in URL, got %q", snap.upgradeURL)
		}
	})

	t.Run("mulaw_format_maps_to_mulaw_8000", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:       strings.NewReader("a"),
			AudioFormat: "audio/mulaw",
			Stream:      true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "encoding=mulaw") {
			t.Fatalf("expected encoding=mulaw in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "sample_rate=8000") {
			t.Fatalf("expected sample_rate=8000 in URL, got %q", snap.upgradeURL)
		}
	})

	t.Run("opus_format_maps_to_opus_no_sample_rate", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:       strings.NewReader("a"),
			AudioFormat: "audio/opus",
			Stream:      true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "encoding=opus") {
			t.Fatalf("expected encoding=opus in URL, got %q", snap.upgradeURL)
		}
		if strings.Contains(snap.upgradeURL, "sample_rate=") {
			t.Fatalf("did not expect sample_rate for opus, got %q", snap.upgradeURL)
		}
	})

	t.Run("custom_model_and_language_forwarded", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:    strings.NewReader("a"),
			Model:    "nova-2-medical",
			Language: "fr",
			Stream:   true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "model=nova-2-medical") {
			t.Fatalf("expected model in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "language=fr") {
			t.Fatalf("expected language=fr in URL, got %q", snap.upgradeURL)
		}
	})

	t.Run("audio_bytes_forwarded_as_binary_frames", func(t *testing.T) {
		payload := strings.Repeat("X", 9_000) // > one chunk, < two chunks
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader(payload),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		var total int
		for _, frame := range snap.binaryFrames {
			total += len(frame)
			if len(frame) > liveAudioChunkBytes {
				t.Fatalf("binary frame %d bytes exceeded chunk cap %d", len(frame), liveAudioChunkBytes)
			}
		}
		if total != len(payload) {
			t.Fatalf("expected %d total bytes streamed, got %d", len(payload), total)
		}
	})

	t.Run("close_stream_sent_after_audio_eof", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !snap.closeReceived {
			t.Fatalf("expected CloseStream sentinel text frame, got %+v", snap.textFrames)
		}
	})

	t.Run("non_results_frames_forwarded_with_type", func(t *testing.T) {
		// Use the REAL upstream payload shapes. UtteranceEnd and
		// SpeechStarted both reuse the `channel` key with an array
		// value [0, 1] (channel indices), not an object. A previous
		// regression decoded `channel` into a struct-typed Channel
		// field on the envelope, failing the whole frame before the
		// Type guard could dispatch — see Issue #9.
		mixed := []string{
			`{"type":"Metadata","request_id":"r1","transaction_key":"deprecated","sha256":"x","created":"2026-05-17T00:00:00.000Z","duration":0,"channels":1}`,
			`{"type":"SpeechStarted","channel":[0,1],"timestamp":0.1}`,
			resultFrame("hi", 0.0, 0.5, false, nil),
			`{"type":"UtteranceEnd","channel":[0,1],"last_word_end":0.5}`,
			resultFrame("hi there", 0.0, 1.0, true, nil),
		}
		p, _, done := newLiveTestServer(t, liveServerOpts{}, mixed)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		<-done
		// All 5 frames now arrive; consumers dispatch on Type.
		if len(segs) != 5 {
			t.Fatalf("expected 5 segments (all event types forwarded), got %d", len(segs))
		}
		wantTypes := []string{"Metadata", "SpeechStarted", "Results", "UtteranceEnd", "Results"}
		for i, want := range wantTypes {
			if segs[i].Type != want {
				t.Errorf("segs[%d].Type = %q, want %q", i, segs[i].Type, want)
			}
		}
		// Non-Results segments carry no transcript fields.
		for i, idx := range []int{0, 1, 3} {
			if segs[idx].Text != "" {
				t.Errorf("segs[%d].Text = %q, want empty for %s", idx, segs[idx].Text, wantTypes[i])
			}
			if segs[idx].Final {
				t.Errorf("segs[%d].Final true for non-Results event", idx)
			}
		}
		// Results segments carry text + Final.
		if segs[2].Text != "hi" || segs[2].Final {
			t.Errorf("interim Results: text=%q final=%v", segs[2].Text, segs[2].Final)
		}
		if segs[4].Text != "hi there" || !segs[4].Final {
			t.Errorf("final Results: text=%q final=%v", segs[4].Text, segs[4].Final)
		}
		// Raw populated on every segment.
		for i, s := range segs {
			if len(s.Raw) == 0 {
				t.Errorf("segs[%d].Raw is empty", i)
			}
		}
	})

	t.Run("speech_final_distinct_from_final", func(t *testing.T) {
		// Interim Results with speech_final=true is meaningful even when
		// is_final=false: the model has detected end-of-utterance.
		frames := []string{
			`{"type":"Results","start":0.0,"duration":0.5,"is_final":false,"speech_final":true,"channel":{"alternatives":[{"transcript":"hi","confidence":0.95,"words":[]}]}}`,
			`{"type":"Results","start":0.0,"duration":1.0,"is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"hi there","confidence":0.99,"words":[]}]}}`,
		}
		p, _, done := newLiveTestServer(t, liveServerOpts{}, frames)
		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		<-done
		if len(segs) != 2 {
			t.Fatalf("expected 2 segments, got %d", len(segs))
		}
		// First: interim transcript with end-of-speech signal (turn-taking trigger).
		if segs[0].Final {
			t.Error("seg[0].Final should be false (interim)")
		}
		if !segs[0].SpeechFinal {
			t.Error("seg[0].SpeechFinal should be true")
		}
		// Second: final transcript.
		if !segs[1].Final {
			t.Error("seg[1].Final should be true")
		}
		if !segs[1].SpeechFinal {
			t.Error("seg[1].SpeechFinal should be true")
		}
	})

	t.Run("4xx_handshake_returns_err_upstream", func(t *testing.T) {
		p, _, _ := newLiveTestServer(t, liveServerOpts{rejectHandshake: http.StatusUnauthorized}, nil)

		_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err == nil {
			t.Fatal("expected error from 401 handshake")
		}
		var upstream *llmrouter.ErrUpstream
		if !errors.As(err, &upstream) {
			t.Fatalf("expected *llmrouter.ErrUpstream, got %T (%v)", err, err)
		}
		if upstream.Provider != "deepgram" {
			t.Fatalf("expected provider deepgram, got %q", upstream.Provider)
		}
		if upstream.StatusCode != 0 {
			t.Fatalf("expected StatusCode 0 on dial failure, got %d", upstream.StatusCode)
		}
	})

	t.Run("context_cancel_mid_stream_closes_cleanly", func(t *testing.T) {
		// Server holds the connection open and delays the final frame
		// so we have a window to cancel.
		p, _, done := newLiveTestServer(t, liveServerOpts{
			pauseBeforeFinal: 2 * time.Second,
			holdOpen:         true,
		}, frames)

		ctx, cancel := context.WithCancel(context.Background())
		stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
			Audio:  strings.NewReader(strings.Repeat("y", 8000)),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Read at least one interim segment to confirm the pipe is alive.
		seg, ok := <-stream.Segments()
		if !ok {
			t.Fatal("expected at least one segment before cancel")
		}
		_ = seg

		cancel()

		// Drain remaining segments; the channel must close.
		for range stream.Segments() {
		}
		_ = stream.Err()
		<-done
	})

	t.Run("missing_audio_reader_errors", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("dg-test"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, err = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Stream: true,
		})
		if err == nil {
			t.Fatal("expected error when Audio is nil")
		}
	})

	t.Run("final_segment_carries_raw_payload", func(t *testing.T) {
		p, _, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		<-done
		final := segs[len(segs)-1]
		if len(final.Raw) == 0 {
			t.Fatal("expected Raw populated on final segment")
		}
		var anything any
		if err := json.Unmarshal(final.Raw, &anything); err != nil {
			t.Fatalf("Raw is not valid JSON: %v", err)
		}
	})

	t.Run("final_segment_confidence_and_timings", func(t *testing.T) {
		p, _, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		<-done
		final := segs[len(segs)-1]
		if diff := final.Confidence - 0.97; diff > 0.001 || diff < -0.001 {
			t.Fatalf("expected confidence ~0.97, got %v", final.Confidence)
		}
		// start=0.0, duration=1.04 → End = 1040ms
		if final.End != 1040*time.Millisecond {
			t.Fatalf("expected End=1040ms, got %v", final.End)
		}
		if final.Start != 0 {
			t.Fatalf("expected Start=0, got %v", final.Start)
		}
	})

	t.Run("raw_endpointing_passthrough_overrides_default", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
			Raw:    json.RawMessage(`{"endpointing": 500, "vad_events": true}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "endpointing=500") {
			t.Fatalf("expected endpointing=500 in URL, got %q", snap.upgradeURL)
		}
		if !strings.Contains(snap.upgradeURL, "vad_events=true") {
			t.Fatalf("expected vad_events=true in URL, got %q", snap.upgradeURL)
		}
	})

	t.Run("raw_diarize_batch_key_also_works_streaming", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
			Raw:    json.RawMessage(`{"diarize": true}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if !strings.Contains(snap.upgradeURL, "diarize=true") {
			t.Fatalf("expected diarize=true in URL, got %q", snap.upgradeURL)
		}
	})

	t.Run("words_emitted_on_final_segment", func(t *testing.T) {
		p, _, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		<-done
		final := segs[len(segs)-1]
		if len(final.Words) != 2 {
			t.Fatalf("expected 2 words on final, got %d", len(final.Words))
		}
		if final.Words[0].Word != "Hello" {
			t.Fatalf("expected punctuated 'Hello', got %q", final.Words[0].Word)
		}
		if final.Words[0].Start != 0 {
			t.Fatalf("expected first word start=0, got %v", final.Words[0].Start)
		}
		if final.Words[0].End != 500*time.Millisecond {
			t.Fatalf("expected first word end=500ms, got %v", final.Words[0].End)
		}
	})

	t.Run("wss_scheme_used_for_https_base", func(t *testing.T) {
		// Build URL directly: don't need a server.
		urlStr, err := buildLiveURL("https://api.deepgram.com", llmrouter.TranscribeRequest{Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.HasPrefix(urlStr, "wss://") {
			t.Fatalf("expected wss:// prefix, got %q", urlStr)
		}
	})

	t.Run("ws_scheme_used_for_http_base", func(t *testing.T) {
		urlStr, err := buildLiveURL("http://localhost:1234", llmrouter.TranscribeRequest{Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.HasPrefix(urlStr, "ws://") {
			t.Fatalf("expected ws:// prefix, got %q", urlStr)
		}
	})

	t.Run("multiple_audio_chunks_streamed_in_order", func(t *testing.T) {
		// Build distinguishable contents per chunk.
		var payloadBuilder strings.Builder
		for i := 0; i < 3; i++ {
			payloadBuilder.WriteString(strings.Repeat(fmt.Sprintf("%d", i), 100))
		}
		payload := payloadBuilder.String()
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader(payload),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		// Reassemble in receive order and confirm it matches the input.
		var reassembled strings.Builder
		for _, f := range snap.binaryFrames {
			reassembled.Write(f)
		}
		if reassembled.String() != payload {
			t.Fatalf("reassembled audio mismatch: want %d bytes, got %d", len(payload), reassembled.Len())
		}
	})

	t.Run("malformed_results_frame_terminates_stream_with_error", func(t *testing.T) {
		p, _, done := newLiveTestServer(t, liveServerOpts{}, []string{
			`{not valid json`,
		})

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Drain may produce zero segments; we only care about Err().
		for range stream.Segments() {
		}
		if stream.Err() == nil {
			t.Fatal("expected non-nil terminal error from malformed JSON")
		}
		<-done
	})

	t.Run("stream_false_uses_batch_path_unchanged", func(t *testing.T) {
		// Spin up a plain HTTP server that returns the canned sample
		// response and assert: it received a POST to /v1/listen, the
		// stream came back with one Final segment.
		var captured *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &captured, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: false,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if captured == nil {
			t.Fatal("expected batch HTTP path to be hit")
		}
		if captured.URL.Path != "/v1/listen" || captured.Method != http.MethodPost {
			t.Fatalf("expected POST /v1/listen on batch path, got %s %s", captured.Method, captured.URL.Path)
		}
		if len(segs) != 1 || !segs[0].Final {
			t.Fatalf("expected 1 Final segment from batch path, got %+v", segs)
		}
	})

	t.Run("encoding_for_unknown_format_omits_encoding", func(t *testing.T) {
		p, cap, done := newLiveTestServer(t, liveServerOpts{}, frames)

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:       strings.NewReader("a"),
			AudioFormat: "audio/webm",
			Stream:      true,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		<-done
		snap := cap.snapshot()
		if strings.Contains(snap.upgradeURL, "encoding=") {
			t.Fatalf("expected no encoding= for unknown format, got %q", snap.upgradeURL)
		}
	})
}

// TestDecodeLiveFrame_VariantChannelShapes locks down Issue #9: the
// `channel` key has different shapes in different upstream frame types
// (object for Results, array for UtteranceEnd / SpeechStarted). A
// previous regression unmarshalled `channel` into a struct-typed field
// on the envelope, dropping non-Results frames at the JSON layer.
func TestDecodeLiveFrame_VariantChannelShapes(t *testing.T) {
	cases := []struct {
		name        string
		payload     string
		wantType    string
		wantText    string
		wantFinal   bool
		wantSpeechF bool
	}{
		{
			name:     "UtteranceEnd_with_channel_array",
			payload:  `{"type":"UtteranceEnd","channel":[0,1],"last_word_end":2.42}`,
			wantType: "UtteranceEnd",
		},
		{
			name:     "SpeechStarted_with_channel_array",
			payload:  `{"type":"SpeechStarted","channel":[0,1],"timestamp":0.42}`,
			wantType: "SpeechStarted",
		},
		{
			name:     "Metadata_no_channel",
			payload:  `{"type":"Metadata","request_id":"r1","duration":1.5,"channels":1}`,
			wantType: "Metadata",
		},
		{
			name:        "Results_with_object_channel",
			payload:     `{"type":"Results","start":0.0,"duration":1.0,"is_final":true,"speech_final":true,"channel":{"alternatives":[{"transcript":"hello world","confidence":0.97,"words":[]}]}}`,
			wantType:    "Results",
			wantText:    "hello world",
			wantFinal:   true,
			wantSpeechF: true,
		},
		{
			name:     "Results_with_unexpected_channel_shape_degrades",
			payload:  `{"type":"Results","start":0.0,"duration":0.5,"is_final":false,"channel":"weird-future-shape"}`,
			wantType: "Results",
		},
		{
			name:     "Results_with_empty_alternatives",
			payload:  `{"type":"Results","start":0.0,"duration":0.5,"channel":{"alternatives":[]}}`,
			wantType: "Results",
		},
		{
			name:     "Future_unknown_event_type_with_channel_array",
			payload:  `{"type":"SomeFutureEvent","channel":[0,1,2],"foo":"bar"}`,
			wantType: "SomeFutureEvent",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seg, ok, err := decodeLiveFrame([]byte(tc.payload))
			if err != nil {
				t.Fatalf("decodeLiveFrame returned error for %s payload: %v", tc.name, err)
			}
			if !ok {
				t.Fatalf("decodeLiveFrame returned ok=false; expected forwarded segment")
			}
			if seg.Type != tc.wantType {
				t.Errorf("Type = %q, want %q", seg.Type, tc.wantType)
			}
			if seg.Text != tc.wantText {
				t.Errorf("Text = %q, want %q", seg.Text, tc.wantText)
			}
			if seg.Final != tc.wantFinal {
				t.Errorf("Final = %v, want %v", seg.Final, tc.wantFinal)
			}
			if seg.SpeechFinal != tc.wantSpeechF {
				t.Errorf("SpeechFinal = %v, want %v", seg.SpeechFinal, tc.wantSpeechF)
			}
			if len(seg.Raw) == 0 {
				t.Errorf("Raw must carry original bytes")
			}
		})
	}
}

// TestDecodeLiveFrame_MalformedEnvelopeStillErrors confirms that
// genuinely malformed JSON (not just a variant-shape `channel`) still
// surfaces as an error rather than being swallowed.
func TestDecodeLiveFrame_MalformedEnvelopeStillErrors(t *testing.T) {
	_, _, err := decodeLiveFrame([]byte(`{not json at all`))
	if err == nil {
		t.Fatal("expected error from malformed envelope JSON")
	}
}
