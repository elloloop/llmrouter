package elevenlabs

import "testing"

// TestHTTPToWSScheme covers all branches of the scheme normaliser used to
// translate REST base URLs into their WebSocket equivalents. Pure helper
// — no I/O, no fixtures.
func TestHTTPToWSScheme(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"https_to_wss", "https://api.elevenlabs.io", "wss://api.elevenlabs.io"},
		{"http_to_ws", "http://localhost:8080", "ws://localhost:8080"},
		{"wss_unchanged", "wss://api.elevenlabs.io", "wss://api.elevenlabs.io"},
		{"ws_unchanged", "ws://localhost:8080", "ws://localhost:8080"},
		{"empty_unchanged", "", ""},
		{"unknown_scheme_passthrough", "tcp://host:1", "tcp://host:1"},
		{"trailing_slash_preserved_https", "https://api.elevenlabs.io/", "wss://api.elevenlabs.io/"},
		{"trailing_slash_preserved_http", "http://localhost/", "ws://localhost/"},
		{"path_preserved", "https://api.elevenlabs.io/v1/path", "wss://api.elevenlabs.io/v1/path"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := httpToWSScheme(tc.in)
			if got != tc.want {
				t.Errorf("httpToWSScheme(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
