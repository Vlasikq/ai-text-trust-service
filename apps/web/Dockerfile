# Multi-stage сборка Next.js standalone — итоговый образ ~150 MB.
# Стандартный паттерн из официальной документации Next.js.

# ── Stage 1: deps ──────────────────────────────────────────────
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund --ignore-scripts \
    && npm rebuild sharp

# ── Stage 2: build ─────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# В same-origin prod-режиме API_URL пустой — fetch идёт на тот же origin.
# Если нужно cross-origin — переопределить --build-arg NEXT_PUBLIC_API_URL=https://api.example.com.
ARG NEXT_PUBLIC_API_URL=""
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build

# ── Stage 3: runner ────────────────────────────────────────────
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

RUN addgroup -g 1001 nodejs && adduser -S -u 1001 -G nodejs nextjs

# Standalone-output содержит только нужные node_modules + минимальный server.js.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000

CMD ["node", "server.js"]
