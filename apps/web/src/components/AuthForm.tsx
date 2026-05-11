"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const schema = z.object({
  email: z.email({ message: "Некорректный email" }),
  password: z
    .string()
    .min(8, "Минимум 8 символов")
    .max(128, "Максимум 128 символов"),
});

type FormValues = z.infer<typeof schema>;

interface Props {
  mode: "login" | "register";
}

export function AuthForm({ mode }: Props) {
  const { signIn, signUp } = useAuth();
  const router = useRouter();
  const [serverError, setServerError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { email: "", password: "" },
  });

  const onSubmit = handleSubmit(async (data) => {
    setServerError(null);
    try {
      if (mode === "login") {
        await signIn(data.email, data.password);
      } else {
        await signUp(data.email, data.password);
      }
      router.push("/me/history");
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 409) setServerError("Email уже зарегистрирован");
        else if (e.status === 401)
          setServerError("Неверный email или пароль");
        else if (e.status === 429)
          setServerError(
            mode === "login"
              ? "Слишком много попыток входа. Попробуйте через минуту."
              : "Слишком много регистраций с этого IP. Попробуйте позже.",
          );
        else setServerError(e.detail);
      } else {
        setServerError("Ошибка соединения. Попробуйте позже.");
      }
    }
  });

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      <label className="block">
        <span className="block text-sm font-medium mb-1">Email</span>
        <input
          type="email"
          autoComplete={mode === "login" ? "username" : "email"}
          {...register("email")}
          className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
        />
        {errors.email && (
          <span className="text-xs text-[var(--danger)] mt-1 block">
            {errors.email.message}
          </span>
        )}
      </label>

      <label className="block">
        <span className="block text-sm font-medium mb-1">Пароль</span>
        <input
          type="password"
          autoComplete={
            mode === "login" ? "current-password" : "new-password"
          }
          {...register("password")}
          className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
        />
        {errors.password && (
          <span className="text-xs text-[var(--danger)] mt-1 block">
            {errors.password.message}
          </span>
        )}
      </label>

      {serverError && (
        <div className="p-3 rounded-md bg-red-50 border border-red-200 text-[var(--danger)] text-sm dark:bg-red-950/30">
          {serverError}
        </div>
      )}

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full px-4 py-2.5 rounded-md bg-[var(--primary)] text-[var(--primary-fg)] font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {isSubmitting
          ? "Подождите..."
          : mode === "login"
          ? "Войти"
          : "Зарегистрироваться"}
      </button>
    </form>
  );
}
