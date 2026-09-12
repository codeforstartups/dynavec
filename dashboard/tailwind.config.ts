import type { Config } from "tailwindcss";

// dynavec brand tokens — kept in sync with the landing page (styles.css).
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#fbfaf8",
        surface: "#ffffff",
        ink: "#14110f",
        muted: "#6f6862",
        faint: "#a99f97",
        line: "#ece6df",
        accent: "#e8623b",
        "accent-ink": "#b8472a",
        "accent-soft": "#fdeee8",
        ok: "#2f7d5b",
        err: "#b8472a",
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: "0 6px 30px rgba(20,17,15,.07)",
      },
      borderRadius: {
        xl2: "14px",
      },
    },
  },
  plugins: [],
};
export default config;
