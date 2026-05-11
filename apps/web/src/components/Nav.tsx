"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export function Nav() {
  const { user, signOut, loading } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  const linkCls = (href: string) =>
    `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
      pathname === href
        ? "bg-[var(--primary)] text-[var(--primary-fg)]"
        : "hover:bg-[var(--border)]"
    }`;
  const ariaCurrent = (href: string) =>
    pathname === href ? ("page" as const) : undefined;

  return (
    <header className="border-b border-[var(--border)] bg-[var(--background)] sticky top-0 z-10">
      <div className="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between gap-2">
        <Link href="/" className="font-semibold text-lg shrink-0">
          AI Trust Service
        </Link>
        <nav className="flex items-center gap-1 flex-wrap">
          <Link href="/" className={linkCls("/")} aria-current={ariaCurrent("/")}>
            Анализ
          </Link>
          {user && (
            <Link
              href="/me/history"
              className={linkCls("/me/history")}
              aria-current={ariaCurrent("/me/history")}
            >
              История
            </Link>
          )}
          {user && (
            <Link
              href="/me/stats"
              className={linkCls("/me/stats")}
              aria-current={ariaCurrent("/me/stats")}
            >
              Статистика
            </Link>
          )}
          {user && (
            <Link
              href="/batch"
              className={linkCls("/batch")}
              aria-current={ariaCurrent("/batch")}
            >
              Batch
            </Link>
          )}
          {!loading && !user && (
            <>
              <Link
                href="/login"
                className={linkCls("/login")}
                aria-current={ariaCurrent("/login")}
              >
                Вход
              </Link>
              <Link
                href="/register"
                className={linkCls("/register")}
                aria-current={ariaCurrent("/register")}
              >
                Регистрация
              </Link>
            </>
          )}
          {user && (
            <button
              onClick={async () => {
                await signOut();
                router.push("/");
              }}
              className="px-3 py-1.5 rounded-md text-sm font-medium hover:bg-[var(--border)] text-[var(--muted)]"
            >
              Выйти
            </button>
          )}
        </nav>
      </div>
    </header>
  );
}
