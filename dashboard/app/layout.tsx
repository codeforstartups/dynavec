import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "dynavec · Observability",
  description: "Real-time retrieval observability for dynavec — latency, cache, traces.",
};

// Set the theme class before first paint so there is no light→dark flash.
// Precedence: ?theme= override (deep-linkable, used for parity screenshots) >
// saved choice > OS preference; defaults to light (matching the site).
const themeInit = `(function(){try{var q=new URLSearchParams(location.search).get('theme');var s=localStorage.getItem('dynavec-theme');var d=q?q==='dark':(s?s==='dark':matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.classList.toggle('dark',d);if(q){try{localStorage.setItem('dynavec-theme',d?'dark':'light');}catch(e){}}}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
