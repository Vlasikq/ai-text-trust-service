import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";

export const metadata = { title: "Регистрация — AI Text Trust" };

export default function RegisterPage() {
  return (
    <div className="max-w-sm mx-auto space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Регистрация</h1>
        <p className="text-sm text-[var(--muted)] mt-1">
          Уже есть аккаунт?{" "}
          <Link href="/login" className="underline">
            Войти
          </Link>
          .
        </p>
      </header>
      <AuthForm mode="register" />
      <p className="text-xs text-[var(--muted)]">
        После регистрации все ваши анализы будут сохраняться в личной истории.
        Email не передаётся третьим лицам.
      </p>
    </div>
  );
}
