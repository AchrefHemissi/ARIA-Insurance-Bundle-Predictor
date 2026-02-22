"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Activity, Brain, Upload, BarChart3 } from "lucide-react";

const links = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/predict", label: "Predict", icon: Brain },
  { href: "/batch", label: "Batch", icon: Upload },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass" role="navigation" aria-label="Main navigation">
      <div className="mx-auto max-w-7xl flex items-center justify-between px-6 py-3">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-full bg-neon-cyan/20 flex items-center justify-center glow-cyan">
            <div className="w-3 h-3 rounded-full bg-neon-cyan" />
          </div>
          <span className="font-mono text-sm font-bold tracking-wider text-foreground">
            DATAQUEST<span className="text-neon-cyan">.AI</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                  isActive
                    ? "bg-neon-cyan/10 text-neon-cyan"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
