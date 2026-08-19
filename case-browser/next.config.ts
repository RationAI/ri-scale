import type { NextConfig } from "next";
import { normaliseBasePath } from "./lib/basePath";

/** Prefix the whole app with a path segment, for hosting it under something
 * like example.com/ri-scale rather than at a domain root. Build-time only:
 * Next bakes it into the emitted asset URLs, so changing it means rebuilding.
 *
 * The reverse proxy in front must pass the prefix through, NOT strip it -
 * with basePath set, Next expects to receive /ri-scale/browse. In Traefik that
 * means no stripPrefix middleware on the router. */
const nextConfig: NextConfig = {
  // Emit .next/standalone with a traced, minimal node_modules so the Docker
  // runner stage doesn't have to carry the full dependency tree.
  output: "standalone",
  basePath: normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH),
};

export default nextConfig;
