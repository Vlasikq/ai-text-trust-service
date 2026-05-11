import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "AI Text Trust",
    short_name: "AI Trust",
    description:
      "Сервис для определения текстов, сгенерированных ИИ. Русскоязычная модель.",
    start_url: "/",
    display: "standalone",
    background_color: "#fafafa",
    theme_color: "#2563eb",
    orientation: "portrait",
    lang: "ru",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "maskable" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
