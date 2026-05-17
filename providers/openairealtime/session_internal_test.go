package openairealtime

import "testing"

// TestNormaliseWSScheme covers all branches of the scheme normaliser used
// to translate REST base URLs into their WebSocket equivalents.
func TestNormaliseWSScheme(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"https_to_wss", "https://api.openai.com", "wss://api.openai.com"},
		{"http_to_ws", "http://localhost:8080", "ws://localhost:8080"},
		{"wss_unchanged", "wss://api.openai.com", "wss://api.openai.com"},
		{"ws_unchanged", "ws://localhost:8080", "ws://localhost:8080"},
		{"empty_unchanged", "", ""},
		{"unknown_scheme_passthrough", "tcp://host:1", "tcp://host:1"},
		{"trailing_slash_preserved_https", "https://api.openai.com/", "wss://api.openai.com/"},
		{"path_preserved", "https://api.openai.com/v1/realtime", "wss://api.openai.com/v1/realtime"},
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
