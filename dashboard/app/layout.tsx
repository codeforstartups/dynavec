import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "dynavec · Observability",
  description: "Real-time retrieval observability for dynavec — latency, cache, traces.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
