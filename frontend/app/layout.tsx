import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import Link from "next/link";
import NavLinks from "@/components/nav-links";
import "./globals.css";

const sans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "CFB Predictor",
  description: "Weekly college football win-probability predictions.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="flex min-h-screen flex-col antialiased">
        <header className="bg-canvas">
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-6 px-4 py-4 sm:px-6">
            <Link
              href="/"
              className="text-xl font-bold uppercase tracking-tight text-ink"
            >
              CFB Predictor
            </Link>
            <NavLinks />
          </div>
        </header>
        <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-10 sm:px-6 sm:py-14">
          {children}
        </div>
        <footer className="mx-auto w-full max-w-5xl px-4 pb-10 font-mono text-[11px] uppercase tracking-wide text-ink-muted sm:px-6">
          Data via <Link href="https://collegefootballdata.com" className="text-accent hover:text-accent-dark">collegefootballdata.com</Link>
        </footer>
      </body>
    </html>
  );
}
