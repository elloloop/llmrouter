package llmrouter

import "context"

// Stream is the streaming-completion handle. Consume Chunks() until it
// closes, then check Err() for any terminal error. Cancelling the
// context passed to CompletionStream causes the stream to close cleanly.
//
// Stream is single-consumer: only one goroutine should read Chunks().
type Stream struct {
	chunks chan Chunk
	cancel context.CancelFunc
	errMu  chan struct{} // closed when err is final
	err    error
}

// newStream constructs a Stream wired to a producer goroutine. The
// producer must call SendChunk for each chunk, then SetErr (with nil on
// success), then Close.
func newStream(cancel context.CancelFunc) *Stream {
	return &Stream{
		chunks: make(chan Chunk, 16),
		cancel: cancel,
		errMu:  make(chan struct{}),
	}
}

// Chunks returns the receive-only chunk channel.
func (s *Stream) Chunks() <-chan Chunk { return s.chunks }

// Err returns the terminal error, if any, after Chunks is drained.
// Blocks until the producer finishes.
func (s *Stream) Err() error {
	<-s.errMu
	return s.err
}

// Cancel asks the producer to stop. The chunks channel will close once
// the producer notices the cancellation. Safe to call multiple times.
func (s *Stream) Cancel() {
	if s.cancel != nil {
		s.cancel()
	}
}

// sendChunk delivers one chunk to the consumer; respects ctx so a
// disconnected consumer doesn't pin the producer.
func (s *Stream) sendChunk(ctx context.Context, c Chunk) bool {
	select {
	case s.chunks <- c:
		return true
	case <-ctx.Done():
		return false
	}
}

// finish closes the stream and finalises the error. Idempotent.
func (s *Stream) finish(err error) {
	s.err = err
	close(s.chunks)
	close(s.errMu)
}

// NewStream is exported for provider implementations in subpackages.
// Returns the Stream plus producer helpers.
func NewStream(parent context.Context) (*Stream, context.Context, ProducerHooks) {
	ctx, cancel := context.WithCancel(parent)
	s := newStream(cancel)
	return s, ctx, ProducerHooks{
		Send:   func(c Chunk) bool { return s.sendChunk(ctx, c) },
		Finish: func(err error) { s.finish(err) },
	}
}

// ProducerHooks are the callbacks a provider uses to feed a Stream.
// Send returns false if the consumer cancelled — the producer should
// then stop and call Finish(ctx.Err()).
type ProducerHooks struct {
	Send   func(Chunk) bool
	Finish func(error)
}
