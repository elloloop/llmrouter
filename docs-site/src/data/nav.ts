// Single source of truth for the documentation nav. Used by:
//   - the sidebar (rendered in DocsLayout)
//   - breadcrumbs
//   - prev/next links at the bottom of every page
//
// Keep `href` paths in sync with the file paths under src/pages/docs/.

export interface NavItem {
  label: string;
  href: string;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

export const BASE = "/llmrouter/docs";

export const sidebarSections: NavSection[] = [
  {
    title: "Getting Started",
    items: [
      { label: "Introduction", href: `${BASE}/` },
      { label: "Installation", href: `${BASE}/installation` },
      { label: "Quick Start", href: `${BASE}/quickstart` },
      { label: "Architecture Overview", href: `${BASE}/architecture/overview` },
    ],
  },
  {
    title: "Concepts",
    items: [
      { label: "Provider Interface", href: `${BASE}/concepts/provider-interface` },
      { label: "Streaming Model", href: `${BASE}/concepts/streaming-model` },
      { label: "Byte Passthrough", href: `${BASE}/concepts/byte-passthrough` },
      { label: "Configuration & Options", href: `${BASE}/concepts/configuration` },
      { label: "Error Handling", href: `${BASE}/concepts/error-handling` },
      { label: "Context & Cancellation", href: `${BASE}/concepts/context-cancellation` },
      { label: "Embeddings", href: `${BASE}/concepts/embeddings` },
      { label: "Audio (TTS + STT)", href: `${BASE}/concepts/audio` },
      { label: "Realtime sessions", href: `${BASE}/concepts/realtime` },
      { label: "Structured outputs", href: `${BASE}/concepts/structured-outputs` },
      { label: "Rerank", href: `${BASE}/concepts/rerank` },
      { label: "The Router", href: `${BASE}/concepts/router` },
    ],
  },
  {
    title: "Providers",
    items: [
      { label: "OpenAI", href: `${BASE}/providers/openai` },
      { label: "OpenAI Realtime", href: `${BASE}/providers/openai-realtime` },
      { label: "Gemini Live", href: `${BASE}/providers/gemini-live` },
      { label: "Anthropic", href: `${BASE}/providers/anthropic` },
      { label: "Azure OpenAI", href: `${BASE}/providers/azure-openai` },
      { label: "Azure Foundry — Anthropic", href: `${BASE}/providers/azure-anthropic` },
      { label: "Azure Foundry — Serverless", href: `${BASE}/providers/azure-serverless` },
      { label: "AWS Bedrock", href: `${BASE}/providers/bedrock` },
      { label: "Google Vertex AI", href: `${BASE}/providers/vertex` },
      { label: "Vertex AI — Anthropic", href: `${BASE}/providers/vertex-anthropic` },
      { label: "Google Gemini (AI Studio)", href: `${BASE}/providers/gemini` },
      { label: "Cohere", href: `${BASE}/providers/cohere` },
      { label: "Mistral", href: `${BASE}/providers/mistral` },
      { label: "OpenAI-compatible Endpoints", href: `${BASE}/providers/openai-compatible` },
      { label: "ElevenLabs", href: `${BASE}/providers/elevenlabs` },
      { label: "Deepgram", href: `${BASE}/providers/deepgram` },
      { label: "Cartesia", href: `${BASE}/providers/cartesia` },
      { label: "Voyage AI", href: `${BASE}/providers/voyage` },
      { label: "Adding a New Provider", href: `${BASE}/providers/adding-a-provider` },
    ],
  },
  {
    title: "Guides",
    items: [
      { label: "Build a Chat Gateway", href: `${BASE}/guides/chat-gateway` },
      { label: "Multi-provider Failover", href: `${BASE}/guides/multi-provider-failover` },
      { label: "Switching Providers at Runtime", href: `${BASE}/guides/switching-providers` },
      { label: "Custom HTTP Client & Retries", href: `${BASE}/guides/custom-http-client` },
      { label: "Cancellation & Timeouts", href: `${BASE}/guides/cancellation-timeouts` },
      { label: "Byte-passthrough Proxy", href: `${BASE}/guides/byte-passthrough-proxy` },
    ],
  },
  {
    title: "API Reference",
    items: [
      { label: "Package llmrouter", href: `${BASE}/api/llmrouter` },
      { label: "Options", href: `${BASE}/api/options` },
      { label: "Stream", href: `${BASE}/api/stream` },
      { label: "Errors", href: `${BASE}/api/errors` },
      { label: "Audio (Speaker + Transcriber)", href: `${BASE}/api/audio` },
      { label: "Embeddings (Embedder)", href: `${BASE}/api/embeddings` },
      { label: "Rerank", href: `${BASE}/api/rerank` },
      { label: "Provider openai", href: `${BASE}/api/provider-openai` },
      { label: "Provider anthropic", href: `${BASE}/api/provider-anthropic` },
      { label: "Package router", href: `${BASE}/api/router` },
    ],
  },
  {
    title: "Project",
    items: [
      { label: "Comparison vs Alternatives", href: `${BASE}/project/comparison` },
      { label: "Roadmap", href: `${BASE}/project/roadmap` },
      { label: "Contributing", href: `${BASE}/project/contributing` },
      { label: "Changelog", href: `${BASE}/project/changelog` },
    ],
  },
];

// Flat ordered list with section labels — drives breadcrumbs and prev/next.
export interface FlatNavItem extends NavItem {
  section: string;
}

export const flatNav: FlatNavItem[] = sidebarSections.flatMap((section) =>
  section.items.map((item) => ({ ...item, section: section.title })),
);

// Normalize a path to match how `href` is defined above (strip trailing slash,
// keep root as just the BASE).
function normalize(p: string): string {
  if (!p) return p;
  if (p === BASE || p === `${BASE}/`) return `${BASE}/`;
  return p.replace(/\/+$/, "");
}

export function findCurrent(currentPath: string): FlatNavItem | undefined {
  const target = normalize(currentPath);
  return flatNav.find(
    (item) => normalize(item.href) === target || item.href === target,
  );
}

export function findPrevNext(currentPath: string): {
  prev?: FlatNavItem;
  next?: FlatNavItem;
} {
  const target = normalize(currentPath);
  const idx = flatNav.findIndex(
    (item) => normalize(item.href) === target || item.href === target,
  );
  if (idx === -1) return {};
  return {
    prev: idx > 0 ? flatNav[idx - 1] : undefined,
    next: idx < flatNav.length - 1 ? flatNav[idx + 1] : undefined,
  };
}

export function buildBreadcrumbs(
  currentPath: string,
): { label: string; href?: string }[] {
  const current = findCurrent(currentPath);
  const docsRoot = { label: "Docs", href: `${BASE}/` };
  if (!current) return [docsRoot];
  if (current.href === `${BASE}/`) return [docsRoot];
  return [
    docsRoot,
    { label: current.section },
    { label: current.label },
  ];
}
