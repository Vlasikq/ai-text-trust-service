import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone: производит .next/standalone/ с минимальным Node-runtime + только нужные deps.
  // Итоговый образ ~150 MB вместо ~1 GB.
  output: "standalone",
};

export default nextConfig;
