package anthropic

import (
	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/voyage"
)

// NewRecommendedEmbedder returns an Embedder that talks to Voyage AI —
// the embedding provider Anthropic recommends for Claude users (Anthropic
// itself does not yet ship an embeddings API).
//
// The voyageAPIKey argument is sent to Voyage; it is unrelated to the
// Anthropic API key you may have configured on *anthropic.Provider. We
// deliberately do NOT make *anthropic.Provider itself satisfy Embedder,
// because that would silently hijack the configured Anthropic key to talk
// to a different vendor — surprising and a footgun for users who expected
// requests to stay within Anthropic.
//
// This is a convenience constructor. For more control (custom HTTP client,
// alternate base URL, etc.) instantiate providers/voyage directly.
//
// Documented reasoning: Anthropic recommends Voyage for embedding workloads
// paired with Claude — see
// https://docs.anthropic.com/en/docs/build-with-claude/embeddings.
//
// Example:
//
//	embedder, err := anthropic.NewRecommendedEmbedder(os.Getenv("VOYAGE_API_KEY"))
//	if err != nil {
//	    // handle error
//	}
//	resp, err := embedder.Embed(ctx, llmrouter.EmbedRequest{
//	    Model:  "voyage-3",
//	    Inputs: []string{"hello"},
//	})
func NewRecommendedEmbedder(voyageAPIKey string, opts ...llmrouter.Option) (llmrouter.Embedder, error) {
	all := append([]llmrouter.Option{llmrouter.WithAPIKey(voyageAPIKey)}, opts...)
	p, err := voyage.New(all...)
	if err != nil {
		// Return a true nil interface (not a typed-nil) so callers can
		// safely check `if emb == nil` after handling the error.
		return nil, err
	}
	return p, nil
}
