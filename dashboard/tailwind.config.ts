import type { Config } from "tailwindcss";

// dynavec brand tokens — kept in sync with the landing page (styles.css).
// Values are driven by CSS variables (see app/globals.css) so a single
// `.dark` class on <html> flips the whole palette. Light is the default,
// matching the landing site; dark uses the site's warm inverted tones.
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        faint: "var(--faint)",
        line: "var(--line)",
        accent: "var(--accent)",
        "accent-ink": "var(--accent-ink)",
        "accent-soft": "var(--accent-soft)",
        ok: "var(--ok)",
        err: "var(--err)",
        track: "var(--track)",
        thead: "var(--thead)",
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: "0 6px 30px var(--shadow)",
      },
      borderRadius: {
        xl2: "14px",
      },
    },
  },
  plugins: [],
};
export default config;
