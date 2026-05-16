package llmrouter_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// ---------- Speaker / Transcriber compile-time interface assertions ----------

type fakeSpeaker struct{}

func (f *fakeSpeaker) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	s, _, hooks := llmrouter.NewAudioStream(ctx)
	go hooks.Finish(nil)
	return s, nil
}

type fakeTranscriber struct{}

func (f *fakeTranscriber) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	s, _, hooks := llmrouter.NewTranscriptStream(ctx)
	go hooks.Finish(nil)
	return s, nil
}

var (
	_ llmrouter.Speaker     = (*fakeSpeaker)(nil)
	_ llmrouter.Transcriber = (*fakeTranscriber)(nil)
)

func TestSpeaker_InterfaceSatisfied(t *testing.T) {
	var s llmrouter.Speaker = &fakeSpeaker{}
	got, err := s.Speak(context.Background(), llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	for range got.Chunks() {
	}
	if err := got.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
}

func TestTranscriber_InterfaceSatisfied(t *testing.T) {
	var s llmrouter.Transcriber = &fakeTranscriber{}
	got, err := s.Transcribe(context.Background(), llmrouter.TranscribeRequest{Model: "whisper-1"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	for range got.Segments() {
	}
	if err := got.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
}

// ---------- SpeechRequest JSON ----------

func TestSpeechRequest_OmitemptyZeroOptional(t *testing.T) {
	req := llmrouter.SpeechRequest{Model: "tts-1", Input: "hi"}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, key := range []string{"voice", "response_format", "speed", "stream"} {
		if strings.Contains(s, key) {
			t.Fatalf("expected %q omitted: %s", key, s)
		}
	}
}

func TestSpeechRequest_AllFieldsPopulated(t *testing.T) {
	speed := 1.5
	req := llmrouter.SpeechRequest{
		Model:  "tts-1-hd",
		Input:  "hello",
		Voice:  "alloy",
		Format: "wav",
		Speed:  &speed,
		Stream: true,
	}
	b, _ := json.Marshal(req)
	for _, want := range []string{`"model":"tts-1-hd"`, `"input":"hello"`, `"voice":"alloy"`, `"response_format":"wav"`, `"speed":1.5`, `"stream":true`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("missing %q in %s", want, b)
		}
	}
}

func TestSpeechRequest_SpeedNilOmitted(t *testing.T) {
	req := llmrouter.SpeechRequest{Model: "m", Input: "x"}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "speed") {
		t.Fatalf("nil Speed must be omitted: %s", b)
	}
}

func TestSpeechRequest_SpeedZeroValuePointerEmitted(t *testing.T) {
	// A non-nil pointer to zero must still be emitted (semantically: explicit 0).
	zero := 0.0
	req := llmrouter.SpeechRequest{Model: "m", Input: "x", Speed: &zero}
	b, _ := json.Marshal(req)
	if !strings.Contains(string(b), `"speed":0`) {
		t.Fatalf("non-nil *Speed=0 should marshal: %s", b)
	}
}

func TestSpeechRequest_RawNotMarshaled(t *testing.T) {
	req := llmrouter.SpeechRequest{
		Model: "m",
		Input: "x",
		Raw:   json.RawMessage(`{"vendor":"thing"}`),
	}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "vendor") {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestSpeechRequest_RoundTrip(t *testing.T) {
	speed := 2.0
	orig := llmrouter.SpeechRequest{
		Model:  "tts-1",
		Input:  "round trip",
		Voice:  "echo",
		Format: "mp3",
		Speed:  &speed,
		Stream: true,
	}
	b, _ := json.Marshal(orig)
	var round llmrouter.SpeechRequest
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Speed == nil || *round.Speed != 2.0 {
		t.Fatalf("Speed mismatch: %#v", round.Speed)
	}
	round.Speed = orig.Speed // pointer-equality irrelevant
	if !reflect.DeepEqual(orig, round) {
		t.Fatalf("round-trip mismatch:\norig:  %#v\nround: %#v", orig, round)
	}
}

// ---------- AudioStream lifecycle ----------

func TestNewAudioStream_ReturnsAllThreeValues(t *testing.T) {
	s, ctx, hooks := llmrouter.NewAudioStream(context.Background())
	if s == nil {
		t.Fatal("nil AudioStream")
	}
	if ctx == nil {
		t.Fatal("nil ctx")
	}
	if hooks.Send == nil || hooks.Finish == nil {
		t.Fatal("hooks not populated")
	}
}

func TestAudioStream_SuccessfulSendAndFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	s.ContentType = "audio/mpeg"

	go func() {
		hooks.Send(llmrouter.AudioChunk{Data: []byte("a")})
		hooks.Send(llmrouter.AudioChunk{Data: []byte("b")})
		hooks.Send(llmrouter.AudioChunk{Data: []byte("c")})
		hooks.Finish(nil)
	}()

	var got []string
	for c := range s.Chunks() {
		got = append(got, string(c.Data))
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
	if len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("got = %v", got)
	}
	if s.ContentType != "audio/mpeg" {
		t.Fatalf("ContentType lost: %q", s.ContentType)
	}
}

func TestAudioStream_FinishWithError(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	want := errors.New("audio boom")
	go func() {
		hooks.Send(llmrouter.AudioChunk{Data: []byte("a")})
		hooks.Finish(want)
	}()
	for range s.Chunks() {
	}
	if got := s.Err(); !errors.Is(got, want) {
		t.Fatalf("Err = %v want %v", got, want)
	}
}

func TestAudioStream_NoChunksOnlyFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	go hooks.Finish(nil)
	n := 0
	for range s.Chunks() {
		n++
	}
	if n != 0 {
		t.Fatalf("expected 0, got %d", n)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
}

func TestAudioStream_CancelStopsProducer(t *testing.T) {
	parent := context.Background()
	s, sctx, hooks := llmrouter.NewAudioStream(parent)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			if !hooks.Send(llmrouter.AudioChunk{Data: []byte("x")}) {
				hooks.Finish(sctx.Err())
				return
			}
		}
		hooks.Finish(nil)
	}()

	got := 0
	for range s.Chunks() {
		got++
		if got == 3 {
			s.Cancel()
		}
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("producer did not return after Cancel")
	}
	if err := s.Err(); err == nil {
		t.Fatal("expected canceled error")
	}
}

func TestAudioStream_ParentContextCancelPropagates(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	s, sctx, hooks := llmrouter.NewAudioStream(parent)

	go func() {
		<-sctx.Done()
		hooks.Finish(sctx.Err())
	}()

	cancel()
	for range s.Chunks() {
	}
	if err := s.Err(); err == nil {
		t.Fatal("expected canceled error")
	}
}

func TestAudioStream_CancelIsIdempotent(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	go hooks.Finish(nil)
	for range s.Chunks() {
	}
	s.Cancel()
	s.Cancel()
	s.Cancel()
}

func TestAudioStream_ErrBlocksUntilFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())

	release := make(chan struct{})
	errCh := make(chan error, 1)
	go func() { errCh <- s.Err() }()
	go func() { <-release; hooks.Finish(errors.New("late")) }()

	select {
	case <-errCh:
		t.Fatal("Err returned before finish")
	case <-time.After(50 * time.Millisecond):
	}

	close(release)
	select {
	case err := <-errCh:
		if err == nil || err.Error() != "late" {
			t.Fatalf("Err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Err did not unblock")
	}
}

func TestAudioStream_ErrCallableMultipleTimes(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	go hooks.Finish(errors.New("x"))
	for range s.Chunks() {
	}
	e1 := s.Err()
	e2 := s.Err()
	if e1 == nil || e1.Error() != "x" || e1 != e2 {
		t.Fatalf("Err repeat mismatch: %v %v", e1, e2)
	}
}

func TestAudioStream_ManyChunksDelivered(t *testing.T) {
	const N = 500
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	go func() {
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.AudioChunk{Data: []byte{byte(i)}}) {
				hooks.Finish(nil)
				return
			}
		}
		hooks.Finish(nil)
	}()
	got := 0
	for range s.Chunks() {
		got++
	}
	if got != N {
		t.Fatalf("got %d want %d", got, N)
	}
}

func TestAudioStream_SendReturnsFalseAfterCancel(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	s.Cancel()

	saw := false
	timeout := time.After(2 * time.Second)
	for {
		select {
		case <-timeout:
			t.Fatal("Send never returned false")
		default:
		}
		if !hooks.Send(llmrouter.AudioChunk{Data: []byte("x")}) {
			saw = true
			hooks.Finish(nil)
			break
		}
	}
	if !saw {
		t.Fatal("Send should return false after Cancel")
	}
	for range s.Chunks() {
	}
}

func TestAudioStream_ChunksReceiveOnly(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	go hooks.Finish(nil)
	var _ <-chan llmrouter.AudioChunk = s.Chunks()
	for range s.Chunks() {
	}
}

func TestAudioStream_ConcurrentCancelDuringConsumption(t *testing.T) {
	const N = 1000
	s, _, hooks := llmrouter.NewAudioStream(context.Background())

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.AudioChunk{Data: []byte("x")}) {
				hooks.Finish(context.Canceled)
				return
			}
		}
		hooks.Finish(nil)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(5 * time.Millisecond)
		s.Cancel()
	}()

	for range s.Chunks() {
	}
	wg.Wait()
	_ = s.Err()
}

func TestAudioStream_ContentTypeSettableBeforeSend(t *testing.T) {
	s, _, hooks := llmrouter.NewAudioStream(context.Background())
	s.ContentType = "audio/wav"
	go func() {
		hooks.Send(llmrouter.AudioChunk{Data: []byte("d")})
		hooks.Finish(nil)
	}()
	for range s.Chunks() {
	}
	if s.ContentType != "audio/wav" {
		t.Fatalf("ContentType = %q", s.ContentType)
	}
}

// ---------- TranscribeRequest JSON ----------

func TestTranscribeRequest_OmitemptyZeroOptional(t *testing.T) {
	req := llmrouter.TranscribeRequest{Model: "whisper-1"}
	b, _ := json.Marshal(req)
	s := string(b)
	for _, key := range []string{"language", "prompt", "response_format", "temperature", "stream"} {
		if strings.Contains(s, key) {
			t.Fatalf("expected %q omitted: %s", key, s)
		}
	}
}

func TestTranscribeRequest_AudioReaderNotMarshaled(t *testing.T) {
	req := llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Audio: bytes.NewReader([]byte("WAVDATA")),
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "WAVDATA") {
		t.Fatalf("Audio bytes leaked: %s", b)
	}
	if strings.Contains(string(b), "Audio") || strings.Contains(string(b), `"audio"`) {
		t.Fatalf("Audio key must not appear: %s", b)
	}
}

func TestTranscribeRequest_AudioFormatNotMarshaled(t *testing.T) {
	req := llmrouter.TranscribeRequest{Model: "whisper-1", AudioFormat: "audio/wav"}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "audio/wav") {
		t.Fatalf("AudioFormat leaked: %s", b)
	}
}

func TestTranscribeRequest_FilenameNotMarshaled(t *testing.T) {
	req := llmrouter.TranscribeRequest{Model: "whisper-1", Filename: "secret.wav"}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "secret.wav") {
		t.Fatalf("Filename leaked: %s", b)
	}
}

func TestTranscribeRequest_RawNotMarshaled(t *testing.T) {
	req := llmrouter.TranscribeRequest{
		Model: "whisper-1",
		Raw:   json.RawMessage(`{"x":1}`),
	}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), `"x":1`) {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestTranscribeRequest_PopulatedFieldsMarshal(t *testing.T) {
	temp := 0.2
	req := llmrouter.TranscribeRequest{
		Model:          "whisper-1",
		Language:       "en",
		Prompt:         "context",
		ResponseFormat: "verbose_json",
		Temperature:    &temp,
		Stream:         true,
	}
	b, _ := json.Marshal(req)
	for _, want := range []string{`"language":"en"`, `"prompt":"context"`, `"response_format":"verbose_json"`, `"temperature":0.2`, `"stream":true`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("missing %q in %s", want, b)
		}
	}
}

func TestTranscribeRequest_TemperatureNilOmitted(t *testing.T) {
	req := llmrouter.TranscribeRequest{Model: "m"}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "temperature") {
		t.Fatalf("nil Temperature must be omitted: %s", b)
	}
}

// ---------- TranscriptSegment JSON ----------

func TestTranscriptSegment_FinalMarshalsAsBool(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true}
	b, _ := json.Marshal(seg)
	if !strings.Contains(string(b), `"final":true`) {
		t.Fatalf("expected final:true in %s", b)
	}
}

func TestTranscriptSegment_FinalFalseStillEmitted(t *testing.T) {
	// Final is bool without omitempty in our spec — always emit.
	seg := llmrouter.TranscriptSegment{Text: "interim", Final: false}
	b, _ := json.Marshal(seg)
	if !strings.Contains(string(b), `"final":false`) {
		t.Fatalf("expected final:false in %s", b)
	}
}

func TestTranscriptSegment_WordsRoundTrip(t *testing.T) {
	orig := llmrouter.TranscriptSegment{
		Text:  "hi there",
		Final: true,
		Start: 100 * time.Millisecond,
		End:   600 * time.Millisecond,
		Words: []llmrouter.TranscriptWord{
			{Word: "hi", Start: 100 * time.Millisecond, End: 200 * time.Millisecond, Confidence: 0.99},
			{Word: "there", Start: 250 * time.Millisecond, End: 600 * time.Millisecond, Confidence: 0.85},
		},
		Confidence: 0.9,
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round llmrouter.TranscriptSegment
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Text != orig.Text || round.Final != orig.Final {
		t.Fatalf("basic fields mismatch")
	}
	if len(round.Words) != 2 || round.Words[0].Word != "hi" || round.Words[1].Word != "there" {
		t.Fatalf("Words round-trip mismatch: %#v", round.Words)
	}
}

func TestTranscriptSegment_ConfidenceOmittedWhenZero(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), "confidence") {
		t.Fatalf("zero Confidence must be omitted: %s", b)
	}
}

func TestTranscriptSegment_RawNotMarshaled(t *testing.T) {
	seg := llmrouter.TranscriptSegment{
		Text: "hi",
		Raw:  json.RawMessage(`{"hidden":1}`),
	}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), "hidden") {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestTranscriptSegment_TimingOmittedWhenZero(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), `"start"`) || strings.Contains(string(b), `"end"`) {
		t.Fatalf("zero timing should be omitted: %s", b)
	}
}

func TestTranscriptSegment_EmptyWordsOmitted(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), "words") {
		t.Fatalf("empty Words should be omitted: %s", b)
	}
}

func TestTranscriptSegment_TypeOmittedWhenEmpty(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), `"type"`) {
		t.Fatalf("empty Type should be omitted: %s", b)
	}
}

func TestTranscriptSegment_TypePopulated(t *testing.T) {
	cases := []string{"Results", "SpeechStarted", "UtteranceEnd", "Metadata"}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			seg := llmrouter.TranscriptSegment{Type: in}
			b, _ := json.Marshal(seg)
			if !strings.Contains(string(b), `"type":"`+in+`"`) {
				t.Fatalf("Type missing: %s", b)
			}
		})
	}
}

func TestTranscriptSegment_SpeechFinalOmittedWhenFalse(t *testing.T) {
	seg := llmrouter.TranscriptSegment{Text: "hi", Final: true, SpeechFinal: false}
	b, _ := json.Marshal(seg)
	if strings.Contains(string(b), "speech_final") {
		t.Fatalf("SpeechFinal false should be omitted: %s", b)
	}
}

func TestTranscriptSegment_SpeechFinalDistinctFromFinal(t *testing.T) {
	// Both bools should round-trip independently.
	cases := []struct {
		name        string
		final       bool
		speechFinal bool
	}{
		{"both-false", false, false},
		{"final-only", true, false},
		{"speech-only", false, true},
		{"both-true", true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seg := llmrouter.TranscriptSegment{Text: "x", Final: tc.final, SpeechFinal: tc.speechFinal}
			b, _ := json.Marshal(seg)
			var got llmrouter.TranscriptSegment
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Final != tc.final {
				t.Errorf("Final mismatch: %v vs %v", got.Final, tc.final)
			}
			if got.SpeechFinal != tc.speechFinal {
				t.Errorf("SpeechFinal mismatch: %v vs %v", got.SpeechFinal, tc.speechFinal)
			}
		})
	}
}

// ---------- TranscriptStream lifecycle ----------

func TestNewTranscriptStream_ReturnsAllThreeValues(t *testing.T) {
	s, ctx, hooks := llmrouter.NewTranscriptStream(context.Background())
	if s == nil || ctx == nil || hooks.Send == nil || hooks.Finish == nil {
		t.Fatal("constructor did not wire up correctly")
	}
}

func TestTranscriptStream_SuccessfulSendAndFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go func() {
		hooks.Send(llmrouter.TranscriptSegment{Text: "hello"})
		hooks.Send(llmrouter.TranscriptSegment{Text: " world", Final: true})
		hooks.Finish(nil)
	}()
	var got []string
	for seg := range s.Segments() {
		got = append(got, seg.Text)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
	if len(got) != 2 || got[0] != "hello" || got[1] != " world" {
		t.Fatalf("segments = %v", got)
	}
}

func TestTranscriptStream_FinishWithError(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	want := errors.New("stt boom")
	go func() {
		hooks.Send(llmrouter.TranscriptSegment{Text: "x"})
		hooks.Finish(want)
	}()
	for range s.Segments() {
	}
	if got := s.Err(); !errors.Is(got, want) {
		t.Fatalf("Err = %v want %v", got, want)
	}
}

func TestTranscriptStream_NoSegmentsOnlyFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go hooks.Finish(nil)
	n := 0
	for range s.Segments() {
		n++
	}
	if n != 0 {
		t.Fatalf("expected 0 got %d", n)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
}

func TestTranscriptStream_CancelStopsProducer(t *testing.T) {
	s, sctx, hooks := llmrouter.NewTranscriptStream(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			if !hooks.Send(llmrouter.TranscriptSegment{Text: "x"}) {
				hooks.Finish(sctx.Err())
				return
			}
		}
		hooks.Finish(nil)
	}()
	got := 0
	for range s.Segments() {
		got++
		if got == 3 {
			s.Cancel()
		}
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("producer hang after Cancel")
	}
	if err := s.Err(); err == nil {
		t.Fatal("expected canceled error")
	}
}

func TestTranscriptStream_ParentContextCancelPropagates(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	s, sctx, hooks := llmrouter.NewTranscriptStream(parent)
	go func() {
		<-sctx.Done()
		hooks.Finish(sctx.Err())
	}()
	cancel()
	for range s.Segments() {
	}
	if err := s.Err(); err == nil {
		t.Fatal("expected canceled error")
	}
}

func TestTranscriptStream_CancelIsIdempotent(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go hooks.Finish(nil)
	for range s.Segments() {
	}
	s.Cancel()
	s.Cancel()
	s.Cancel()
}

func TestTranscriptStream_ErrBlocksUntilFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	release := make(chan struct{})
	errCh := make(chan error, 1)
	go func() { errCh <- s.Err() }()
	go func() { <-release; hooks.Finish(errors.New("late")) }()

	select {
	case <-errCh:
		t.Fatal("Err returned before Finish")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-errCh:
		if err == nil || err.Error() != "late" {
			t.Fatalf("Err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Err did not unblock")
	}
}

func TestTranscriptStream_ErrCallableMultipleTimes(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go hooks.Finish(errors.New("x"))
	for range s.Segments() {
	}
	e1 := s.Err()
	e2 := s.Err()
	if e1 == nil || e1.Error() != "x" || e1 != e2 {
		t.Fatalf("Err repeat mismatch")
	}
}

func TestTranscriptStream_ManySegmentsDelivered(t *testing.T) {
	const N = 500
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go func() {
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.TranscriptSegment{Text: "x"}) {
				hooks.Finish(nil)
				return
			}
		}
		hooks.Finish(nil)
	}()
	got := 0
	for range s.Segments() {
		got++
	}
	if got != N {
		t.Fatalf("got %d want %d", got, N)
	}
}

func TestTranscriptStream_SendReturnsFalseAfterCancel(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	s.Cancel()

	saw := false
	timeout := time.After(2 * time.Second)
	for {
		select {
		case <-timeout:
			t.Fatal("Send never returned false")
		default:
		}
		if !hooks.Send(llmrouter.TranscriptSegment{Text: "x"}) {
			saw = true
			hooks.Finish(nil)
			break
		}
	}
	if !saw {
		t.Fatal("Send should return false after Cancel")
	}
	for range s.Segments() {
	}
}

func TestTranscriptStream_SegmentsReceiveOnly(t *testing.T) {
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go hooks.Finish(nil)
	var _ <-chan llmrouter.TranscriptSegment = s.Segments()
	for range s.Segments() {
	}
}

func TestTranscriptStream_ConcurrentCancelDuringConsumption(t *testing.T) {
	const N = 1000
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.TranscriptSegment{Text: "x"}) {
				hooks.Finish(context.Canceled)
				return
			}
		}
		hooks.Finish(nil)
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(5 * time.Millisecond)
		s.Cancel()
	}()
	for range s.Segments() {
	}
	wg.Wait()
	_ = s.Err()
}

func TestTranscriptStream_NonStreamingProviderSingleFinal(t *testing.T) {
	// Verifies docs: non-streaming providers emit exactly one Final=true segment.
	s, _, hooks := llmrouter.NewTranscriptStream(context.Background())
	go func() {
		hooks.Send(llmrouter.TranscriptSegment{Text: "the entire transcript", Final: true})
		hooks.Finish(nil)
	}()
	var segs []llmrouter.TranscriptSegment
	for seg := range s.Segments() {
		segs = append(segs, seg)
	}
	if len(segs) != 1 || !segs[0].Final {
		t.Fatalf("expected exactly one Final segment, got %#v", segs)
	}
}
