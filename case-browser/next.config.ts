import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit .next/standalone with a traced, minimal node_modules so the Docker
  // runner stage doesn't have to carry the full dependency tree.
  output: "standalone",
};

export default nextConfig;
