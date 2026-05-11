import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";

export const metadata = { title: "Вход — AI Text Trust" };

export default function LoginPage() {
  return (
    <div className="max-w-sm mx-auto space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Вход</h1>
        <p className="text-sm text-[var(--muted)] mt-1">
          Нет аккаунта?{" "}
          <Link href="/register" className="underline">
            Зарегистрируйтесь
          </Link>
          .
        </p>
      </header>
      <AuthForm mode="login" />
    </div>
  );
}
