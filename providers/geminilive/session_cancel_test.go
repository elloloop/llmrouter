package geminilive_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/geminilive"
)

func TestContextCancellationDuringRead(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cancel bool
		want   error
	}{
		{name: "cancel", cancel: true, want: context.Canceled},
		{name: "deadline", want: context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, server, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			transport := http.DefaultTransport.(*http.Transport).Clone()
			t.Cleanup(transport.CloseIdleConnections)
			observer := &readObserver{started: make(chan struct{}, 1)}
			p, err := geminilive.New(
				llmrouter.WithAPIKey("test-key"),
				llmrouter.WithBaseURL(server.URL),
				llmrouter.WithHTTPClient(&http.Client{Transport: readObservingTransport{transport, observer}}),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			sess, err := p.Connect(ctx, geminilive.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer sess.Close()

			// Connect has consumed the setup acknowledgement. With no further
			// server frames, an active body read is the pump waiting for input.
			// Cancel only after that read starts so the loop's early ctx check
			// cannot satisfy this regression test instead.
			observer.waitForRead(t, ctx)
			if tc.cancel {
				cancel()
			}
			drainEvents(t, sess, 7*time.Second)
			if err := sess.Err(); !errors.Is(err, tc.want) {
				t.Errorf("Err() = %v, want %v", err, tc.want)
			}
			select {
			case <-done:
			case <-time.After(2 * time.Second):
				t.Fatal("server connection remained open after cancellation")
			}
		})
	}
}

type readObserver struct {
	mu      sync.Mutex
	active  bool
	started chan struct{}
}

func (o *readObserver) waitForRead(t *testing.T, ctx context.Context) {
	t.Helper()
	for {
		o.mu.Lock()
		active := o.active
		o.mu.Unlock()
		if active {
			return
		}
		select {
		case <-o.started:
		case <-ctx.Done():
			t.Fatalf("pump did not start a read before context ended: %v", ctx.Err())
		}
	}
}

type readObservingTransport struct {
	base     http.RoundTripper
	observer *readObserver
}

func (t readObservingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	response, err := t.base.RoundTrip(r)
	if err != nil {
		return nil, err
	}
	body, ok := response.Body.(io.ReadWriteCloser)
	if !ok {
		_ = response.Body.Close()
		return nil, fmt.Errorf("websocket response body is not writable")
	}
	response.Body = observedReadBody{ReadWriteCloser: body, observer: t.observer}
	return response, nil
}

type observedReadBody struct {
	io.ReadWriteCloser
	observer *readObserver
}

func (b observedReadBody) Read(p []byte) (int, error) {
	b.observer.mu.Lock()
	b.observer.active = true
	b.observer.mu.Unlock()
	select {
	case b.observer.started <- struct{}{}:
	default:
	}
	defer func() {
		b.observer.mu.Lock()
		b.observer.active = false
		b.observer.mu.Unlock()
	}()
	return b.ReadWriteCloser.Read(p)
}
