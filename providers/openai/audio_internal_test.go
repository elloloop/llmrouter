package openai

import "testing"

// TestExtensionForFormat exercises every branch of the audio MIME-type →
// file-extension mapping used when synthesising upload filenames for the
// multipart STT request. Missing branches were the main source of the
// previously-low coverage on this helper.
func TestExtensionForFormat(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		// mp3 family
		{"audio_mpeg", "audio/mpeg", ".mp3"},
		{"audio_mp3", "audio/mp3", ".mp3"},
		{"mp3_short", "mp3", ".mp3"},

		// wav family
		{"audio_wav", "audio/wav", ".wav"},
		{"audio_x_wav", "audio/x-wav", ".wav"},
		{"wav_short", "wav", ".wav"},

		// webm
		{"audio_webm", "audio/webm", ".webm"},
		{"webm_short", "webm", ".webm"},

		// m4a / mp4 family
		{"audio_m4a", "audio/m4a", ".m4a"},
		{"audio_mp4", "audio/mp4", ".m4a"},
		{"audio_x_m4a", "audio/x-m4a", ".m4a"},
		{"m4a_short", "m4a", ".m4a"},

		// flac
		{"audio_flac", "audio/flac", ".flac"},
		{"audio_x_flac", "audio/x-flac", ".flac"},
		{"flac_short", "flac", ".flac"},

		// ogg / opus
		{"audio_ogg", "audio/ogg", ".ogg"},
		{"ogg_short", "ogg", ".ogg"},
		{"audio_opus", "audio/opus", ".opus"},
		{"opus_short", "opus", ".opus"},

		// fallbacks
		{"empty_falls_back_to_mp3", "", ".mp3"},
		{"unknown_falls_back_to_mp3", "audio/unknown-codec", ".mp3"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := extensionForFormat(tc.in)
			if got != tc.want {
				t.Errorf("extensionForFormat(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
