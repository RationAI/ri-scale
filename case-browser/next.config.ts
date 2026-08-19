import type { NextConfig } from "next";

/** Prefix the whole app with a path segment, for hosting it under something
 * like example.com/ri-scale rather than at a domain root. Build-time only:
 * Next bakes it into the emitted asset URLs, so changing it means rebuilding.
 * Must be "/something" - a trailing slash or a bare "" breaks the build, so
 * normalise both away here rather than making the caller get it exactly right.
 *
 * The reverse proxy in front must pass the prefix through, NOT strip it -
 * with basePath set, Next expects to receive /ri-scale/browse. In nginx that
 * means `proxy_pass http://host:3000;` with no trailing slash. */
function normaliseBasePath(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim().replace(/\/+$/, "");
  if (!trimmed) return undefined;
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

const nextConfig: NextConfig = {
  // Emit .next/standalone with a traced, minimal node_modules so the Docker
  // runner stage doesn't have to carry the full dependency tree.
  output: "standalone",
  basePath: normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH),
};

export default nextConfig;
