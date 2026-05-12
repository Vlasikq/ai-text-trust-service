"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import {
  apiEndpoints,
  getAccessToken,
  getRefreshToken,
  setAccessToken,
  setRefreshToken,
  subscribeAccess,
  type UserPublic,
} from "@/lib/api";
import { clearSessionHistory } from "@/components/AnalyzeForm";

interface AuthState {
  user: UserPublic | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserPublic | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const me = await apiEndpoints.me();
      setUser(me);
    } catch {
      setUser(null);
    }
  }, []);

  // Загрузка: если есть сохранённый refresh — пытаемся восстановить сессию.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = getRefreshToken();
      if (!stored) {
        setLoading(false);
        return;
      }
      // Дёргаем /me — если access нет, api.ts auto-рефрешнет через сохранённый refresh.
      try {
        const me = await apiEndpoints.me();
        if (!cancelled) setUser(me);
      } catch {
        if (!cancelled) setUser(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Если access потерялся (refresh-cycle провалился) — сбрасываем user.
  useEffect(() => {
    return subscribeAccess((token) => {
      if (!token && !getRefreshToken()) setUser(null);
    });
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const tokens = await apiEndpoints.login(email, password);
    setAccessToken(tokens.access_token);
    setRefreshToken(tokens.refresh_token);
    const me = await apiEndpoints.me();
    setUser(me);
  }, []);

  const signUp = useCallback(async (email: string, password: string) => {
    const r = await apiEndpoints.register(email, password);
    setAccessToken(r.tokens.access_token);
    setRefreshToken(r.tokens.refresh_token);
    setUser(r.user);
  }, []);

  const signOut = useCallback(async () => {
    const r = getRefreshToken();
    if (r) {
      try {
        await apiEndpoints.logout(r);
      } catch {
        // logout идемпотентен на сервере, локально всё равно чистим
      }
    }
    setAccessToken(null);
    setRefreshToken(null);
    clearSessionHistory();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, loading, signIn, signUp, signOut, refresh }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

// Helper for components: returns access if available (forces hook re-render on change).
export function useAccessToken(): string | null {
  const [t, setT] = useState<string | null>(getAccessToken());
  useEffect(() => subscribeAccess(setT), []);
  return t;
}
