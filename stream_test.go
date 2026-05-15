package llmrouter_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

func TestNewStream_ReturnsAllThreeValues(t *testing.T) {
	s, ctx, hooks := llmrouter.NewStream(context.Background())
	if s == nil {
		t.Fatal("nil Stream")
	}
	if ctx == nil {
		t.Fatal("nil ctx")
	}
	if hooks.Send == nil || hooks.Finish == nil {
		t.Fatal("hooks not populated")
	}
}

func TestStream_SuccessfulSendAndFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())

	go func() {
		hooks.Send(llmrouter.Chunk{ID: "a"})
		hooks.Send(llmrouter.Chunk{ID: "b"})
		hooks.Send(llmrouter.Chunk{ID: "c"})
		hooks.Finish(nil)
	}()

	var ids []string
	for c := range s.Chunks() {
		ids = append(ids, c.ID)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
	if len(ids) != 3 || ids[0] != "a" || ids[1] != "b" || ids[2] != "c" {
		t.Fatalf("ids = %v", ids)
	}
}

func TestStream_FinishWithError(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	want := errors.New("boom")
	go func() {
		hooks.Send(llmrouter.Chunk{ID: "a"})
		hooks.Finish(want)
	}()
	for range s.Chunks() {
	}
	if got := s.Err(); !errors.Is(got, want) {
		t.Fatalf("Err = %v, want %v", got, want)
	}
}

func TestStream_NoChunksOnlyFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	go hooks.Finish(nil)
	count := 0
	for range s.Chunks() {
		count++
	}
	if count != 0 {
		t.Fatalf("expected 0 chunks, got %d", count)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
}

func TestStream_CancelStopsProducer(t *testing.T) {
	parent := context.Background()
	s, sctx, hooks := llmrouter.NewStream(parent)

	doneProducing := make(chan struct{})
	go func() {
		defer close(doneProducing)
		for i := 0; i < 1000; i++ {
			if !hooks.Send(llmrouter.Chunk{ID: "x"}) {
				hooks.Finish(sctx.Err())
				return
			}
		}
		hooks.Finish(nil)
	}()

	got := 0
	for c := range s.Chunks() {
		_ = c
		got++
		if got == 3 {
			s.Cancel()
		}
	}

	select {
	case <-doneProducing:
	case <-time.After(2 * time.Second):
		t.Fatal("producer did not return after Cancel")
	}

	if err := s.Err(); err == nil {
		t.Fatal("expected context-canceled error")
	}
}

func TestStream_ParentContextCancelPropagates(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	s, sctx, hooks := llmrouter.NewStream(parent)

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

func TestStream_CancelIsIdempotent(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	go hooks.Finish(nil)
	for range s.Chunks() {
	}
	// Calling Cancel multiple times after the stream finishes must not panic.
	s.Cancel()
	s.Cancel()
	s.Cancel()
}

func TestStream_ErrBlocksUntilFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())

	releaseProducer := make(chan struct{})
	errReceived := make(chan error, 1)

	go func() {
		errReceived <- s.Err() // blocks until finish
	}()

	go func() {
		<-releaseProducer
		hooks.Finish(errors.New("late"))
	}()

	select {
	case <-errReceived:
		t.Fatal("Err() returned before Finish")
	case <-time.After(50 * time.Millisecond):
	}

	close(releaseProducer)
	select {
	case err := <-errReceived:
		if err == nil || err.Error() != "late" {
			t.Fatalf("Err = %v, want 'late'", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Err() did not unblock")
	}
}

func TestStream_ErrCallableMultipleTimesAfterFinish(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	go hooks.Finish(errors.New("x"))
	for range s.Chunks() {
	}
	e1 := s.Err()
	e2 := s.Err()
	e3 := s.Err()
	if e1 == nil || e1.Error() != "x" {
		t.Fatalf("Err[1] = %v", e1)
	}
	if e2 != e1 || e3 != e1 {
		t.Fatalf("Err must return the same error on repeat calls")
	}
}

func TestStream_ManyChunksDelivered(t *testing.T) {
	const N = 500
	s, _, hooks := llmrouter.NewStream(context.Background())
	go func() {
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.Chunk{ID: "x"}) {
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
		t.Fatalf("got %d, want %d", got, N)
	}
}

func TestStream_SendReturnsFalseAfterCancel(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	s.Cancel()

	saw := false
	timeout := time.After(2 * time.Second)
	for {
		select {
		case <-timeout:
			t.Fatal("Send never returned false after Cancel")
		default:
		}
		if !hooks.Send(llmrouter.Chunk{ID: "x"}) {
			saw = true
			hooks.Finish(nil)
			break
		}
	}
	if !saw {
		t.Fatal("Send should return false after cancel")
	}
	// Drain.
	for range s.Chunks() {
	}
}

func TestStream_ChunksReceiveOnlyChannel(t *testing.T) {
	s, _, hooks := llmrouter.NewStream(context.Background())
	go hooks.Finish(nil)
	ch := s.Chunks()
	// Compile-time-ish check: receive should not panic; close should not be allowed.
	var _ <-chan llmrouter.Chunk = ch
	for range ch {
	}
}

func TestStream_ConcurrentCancelDuringConsumption(t *testing.T) {
	const N = 1000
	s, _, hooks := llmrouter.NewStream(context.Background())

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < N; i++ {
			if !hooks.Send(llmrouter.Chunk{ID: "x"}) {
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
