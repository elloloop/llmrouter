package geminilive

import "testing"

// TestNormaliseWSScheme covers all branches of the scheme normaliser used
// to translate REST base URLs into their WebSocket equivalents.
func TestNormaliseWSScheme(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"https_to_wss", "https://generativelanguage.googleapis.com", "wss://generativelanguage.googleapis.com"},
		{"http_to_ws", "http://localhost:8080", "ws://localhost:8080"},
		{"wss_unchanged", "wss://generativelanguage.googleapis.com", "wss://generativelanguage.googleapis.com"},
		{"ws_unchanged", "ws://localhost:8080", "ws://localhost:8080"},
		{"empty_unchanged", "", ""},
		{"unknown_scheme_passthrough", "tcp://host:1", "tcp://host:1"},
		{"trailing_slash_preserved_https", "https://api.example.com/", "wss://api.example.com/"},
		{"path_preserved", "https://api.example.com/v1/live", "wss://api.example.com/v1/live"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := normaliseWSScheme(tc.in)
			if got != tc.want {
				t.Errorf("normaliseWSScheme(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestTruncate covers the debug-string capper used when embedding
// upstream payloads into error messages. Counts in bytes, not runes —
// the limit is byte-based per the implementation.
func TestTruncate(t *testing.T) {
	cases := []struct {
		name string
		in   string
		n    int
		want string
	}{
		{"empty", "", 10, ""},
		{"short_unchanged", "hello", 10, "hello"},
		{"exactly_at_limit", "hello", 5, "hello"},
		{"long_truncated", "abcdefghijklmnop", 5, "abcde..."},
		{"long_truncated_zero", "abc", 0, "..."},
		{"unicode_byte_boundary", "abcdef", 4, "abcd..."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := truncate(tc.in, tc.n)
			if got != tc.want {
				t.Errorf("truncate(%q, %d) = %q, want %q", tc.in, tc.n, got, tc.want)
			}
		})
	}
}
