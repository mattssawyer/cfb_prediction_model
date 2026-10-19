import Link from "next/link";
import NavLinks from "@/components/nav-links";

export default function SiteLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-line bg-panel">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-x-8 gap-y-2 px-5 py-4">
          <Link href="/" className="text-lg font-extrabold tracking-tight">
            CFB Predictor
          </Link>
          <NavLinks />
        </div>
      </header>
      <main className="mx-auto w-full max-w-4xl flex-1 px-5 py-8">{children}</main>
      <footer className="mx-auto w-full max-w-4xl px-5 pb-10 text-sm text-muted">
        Built by{" "}
        <Link href="https://matthewsawyer.dev" className="font-medium text-ink hover:underline">
          Matthew Sawyer
        </Link>
        . Team and game data from{" "}
        <Link href="https://collegefootballdata.com" className="font-medium text-ink hover:underline">
          collegefootballdata.com
        </Link>
        .
      </footer>
    </div>
  );
}
