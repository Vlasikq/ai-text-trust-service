#!/usr/bin/env bash
# Bootstrap-скрипт VM. Запускается из cloud-init runcmd как `bash /opt/aitrust/bootstrap.sh`.
# Strict-режим включён. Шаги, которые ожидаемо могут упасть (systemctl на
# свежем хосте, docker compose до полного запуска), обёрнуты в `|| echo`, чтобы
# bootstrap не валился целиком из-за одного transient-фейла.
set -euo pipefail
set -x

LOG=/var/log/aitrust-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "==> [$(date -Iseconds)] aitrust bootstrap start (uid=$(id -u))"

# 1. YC root cert.
mkdir -p /opt/aitrust/certs
curl -fsSo /opt/aitrust/certs/root.crt https://storage.yandexcloud.net/cloud-certs/CA.pem \
  && echo "==> root cert ok" \
  || echo "==> root cert FAIL ($?)"

# 2. DOMAIN захардкожен render-скриптом.
echo "DOMAIN=__DOMAIN__" > /opt/aitrust/.env
echo "==> domain set to __DOMAIN__"

# 3. Wait until Docker daemon ready (apt install запускает systemd-юнит асинхронно).
systemctl enable docker || echo "==> systemctl enable docker exit=$?"
systemctl start docker  || echo "==> systemctl start docker exit=$?"
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker info >/dev/null 2>&1; then
    echo "==> docker daemon ready (attempt ${i})"
    break
  fi
  echo "==> docker daemon not ready, sleep 2 (attempt ${i})"
  sleep 2
done

# 4. Container Registry публичный (system:allUsers / images.puller),
#    docker login не нужен — pull проходит анонимно.

# 5. Старт стека.
cd /opt/aitrust && docker compose up -d \
  && echo "==> docker compose up issued" \
  || echo "==> docker compose up FAIL ($?)"

# 6. Smoke внутри VM (alembic + warmup + Caddy ACME ~60-90s).
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 15
  if curl -fsS --max-time 5 http://localhost:8000/ready >/dev/null 2>&1; then
    echo "==> api healthy after ${i} attempts"
    break
  fi
  echo "==> api not ready (attempt ${i}/10)"
done

echo "==> bootstrap finished at $(date -Iseconds)"
