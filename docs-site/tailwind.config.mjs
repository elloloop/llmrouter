import { refractionPreset } from "@refraction-ui/tailwind-config";

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}",
    "./node_modules/@refraction-ui/astro/dist/**/*.{astro,js,mjs,ts}",
    "./node_modules/.pnpm/@refraction-ui+astro@*/node_modules/@refraction-ui/astro/dist/**/*.{astro,js,mjs,ts}",
  ],
  presets: [refractionPreset],
  darkMode: "class",
  safelist: ["dark"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
    },
  },
};
