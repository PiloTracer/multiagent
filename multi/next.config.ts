// ./multi/app/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*", // Proxy API requests to your FastAPI backend
      },
    ];
  },
};

export default nextConfig;