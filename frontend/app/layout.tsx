import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sans = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CFB Predictor",
  description: "Weekly college football win-probability predictions.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${mono.variable} antialiased`}
    >
      <body className="min-h-screen bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 font-sans">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
          <nav className="mb-10 flex items-baseline justify-between">
            <Link
              href="/"
              className="font-mono text-sm tracking-wider text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              cfb-predictor
            </Link>
            <Link
              href="/archive"
              className="font-mono text-sm text-neutral-500 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              archive
            </Link>
          </nav>
          {children}
          <footer className="mt-16 pt-6 border-t border-neutral-100 dark:border-neutral-900 text-xs font-mono text-neutral-400 dark:text-neutral-600">
            data:{" "}
            <a
              href="https://collegefootballdata.com"
              className="hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              collegefootballdata.com
            </a>
            {" · "}
            model: LightGBM + isotonic calibration
          </footer>
        </div>
      </body>
    </html>
  );
}
