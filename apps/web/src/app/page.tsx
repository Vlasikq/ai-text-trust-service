"use client";

import Link from "next/link";
import { AnalyzeForm } from "@/components/AnalyzeForm";
import { useAuth } from "@/lib/auth";

export default function Home() {
  const { user, loading } = useAuth();

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          Детекция AI-текстов
        </h1>
        <p className="text-[var(--muted)] text-sm leading-relaxed">
          Сервис анализирует русскоязычный текст и оценивает вероятность того,
          что он сгенерирован ИИ. Результат носит вероятностный характер и не является
          доказательством авторства.
        </p>
      </header>

      {!loading && (
        <div className="text-xs text-[var(--muted)] py-2 px-3 rounded-md border border-[var(--border)] bg-[var(--border)]/30">
          {user ? (
            <>
              Вошли как <strong>{user.email}</strong>. Анализы сохраняются в{" "}
              <Link href="/me/history" className="underline">
                истории
              </Link>
              .
            </>
          ) : (
            <>
              Анонимный режим.{" "}
              <Link href="/login" className="underline">
                Войдите
              </Link>{" "}
              или{" "}
              <Link href="/register" className="underline">
                зарегистрируйтесь
              </Link>
              , чтобы сохранять историю анализов.
            </>
          )}
        </div>
      )}

      <AnalyzeForm />
    </div>
  );
}
