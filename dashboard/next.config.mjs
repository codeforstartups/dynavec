/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export so the dashboard can be served by any static host (incl. the
  // Python `serve()` or GitHub Pages). Data is fetched client-side at runtime.
  output: "export",
  images: { unoptimized: true },
  // Set NEXT_PUBLIC_BASE_PATH="/dynavec/dashboard" when deploying under a subpath.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
  trailingSlash: true,
};
export default nextConfig;
