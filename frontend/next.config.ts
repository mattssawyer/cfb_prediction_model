import path from "node:path";
import type { NextConfig } from "next";

// Mount the app under a URL prefix (e.g. "/cfb-model") when the host domain
// serves other content at the root. Leave empty for local dev / standalone
// deploys. Set NEXT_PUBLIC_BASE_PATH in the Vercel project's env vars.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  // Don't auto-generate AGENTS.md / CLAUDE.md on every build.
  agentRules: false,
  // Fully static export - Vercel/Netlify/any CDN can serve out/ directly.
  output: "export",
  basePath,
  // assetPrefix must match basePath so /_next/* assets resolve correctly
  // when the site is served under a subpath.
  assetPrefix: basePath || undefined,
  // The predictions/ directory sits outside frontend/ in the repo. Include it
  // in the file tracing so `next build` inlines the JSON at build time.
  outputFileTracingIncludes: {
    "/**/*": ["../predictions/**/*"],
  },
  // Static export doesn't run the Image Optimization API.
  images: { unoptimized: true },
  // Pin the workspace root so Turbopack doesn't wander up and pick up
  // ~/bun.lock or similar stray lockfiles.
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
