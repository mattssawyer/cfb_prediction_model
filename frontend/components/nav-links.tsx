"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Picks" },
  { href: "/performance", label: "Track record" },
  { href: "/models", label: "How it works" },
];

export default function NavLinks() {
  const pathname = usePathname();

  return (
    <nav className="flex gap-5 text-sm font-medium">
      {items.map((item) => {
        const active =
          item.href === "/"
            ? pathname === "/" || /^\/\d+\/week\//.test(pathname ?? "")
            : pathname?.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={active ? "text-ink" : "text-muted hover:text-ink"}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
