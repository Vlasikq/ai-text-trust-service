import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { Nav } from "@/components/Nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin", "cyrillic"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin", "cyrillic"],
});

export const metadata: Metadata = {
  title: "AI Text Trust — детекция AI-текстов",
  description:
    "Сервис для определения текстов, сгенерированных ИИ. Русскоязычная модель, каскад TF-IDF + ruRoBERTa.",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "AI Trust",
  },
};

export const viewport: Viewport = {
  themeColor: "#2563eb",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ru"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <AuthProvider>
          <Nav />
          <main className="flex-1 w-full max-w-3xl mx-auto px-4 py-6">
            {children}
          </main>
          <footer className="border-t border-[var(--border)] py-4 text-xs text-[var(--muted)] text-center">
            AI Text Trust · ВКР ВШЭ ФКН ИИ · 2026
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
