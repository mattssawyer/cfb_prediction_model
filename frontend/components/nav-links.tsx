"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Slate" },
  { href: "/performance", label: "Performance" },
  { href: "/models", label: "Models" },
];

export default function NavLinks() {
  const pathname = usePathname();

  return (
    <nav className="flex gap-6 font-mono text-xs uppercase tracking-[0.15em]">
      {items.map((item) => {
        const active =
          item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={
              active
                ? "border-b-2 border-accent pb-1 text-ink"
                : "border-b-2 border-transparent pb-1 text-ink-muted transition-colors hover:text-ink"
            }
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
