import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import expressiveCode from "astro-expressive-code";
import pagefind from "astro-pagefind";

export default defineConfig({
  site: "https://elloloop.github.io",
  base: "/llmrouter",
  outDir: "../docs",
  vite: {
    build: {
      // The repository still keeps a few non-Astro review artifacts under docs/.
      // Do not wipe them when regenerating the GitHub Pages output.
      emptyOutDir: false,
    },
  },
  integrations: [
    expressiveCode({
      themes: ["github-dark-dimmed"],
      styleOverrides: {
        frames: {
          frameBoxShadowCssValue: "none",
          editorTabBarBackground: "hsl(var(--muted))",
          editorActiveTabIndicatorBottomColor: "hsl(var(--primary))",
          editorActiveTabBorderColor: "transparent",
          terminalTitlebarBackground: "hsl(var(--muted))",
          terminalTitlebarBorderBottomColor: "hsl(var(--border))",
          terminalBackground: "#22272e",
          tooltipSuccessBackground: "hsl(var(--primary))",
        },
        borderRadius: "0.5rem",
        codeFontFamily: "var(--font-mono)",
        codeFontSize: "13px",
        codeLineHeight: "1.65",
        uiFontFamily: "var(--font-sans)",
      },
      defaultProps: {
        frame: "code",
      },
      shiki: {},
    }),
    tailwind({ applyBaseStyles: false }),
    pagefind({
      indexConfig: {
        rootSelector: "[data-pagefind-body]",
      },
    }),
  ],
  output: "static",
});
