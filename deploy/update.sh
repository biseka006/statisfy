#!/usr/bin/env bash
# Perbarui Statify: tarik kode terbaru (jika folder ini repo git) lalu build ulang container.
# Pakai: sudo bash deploy/update.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d .git ]]; then
  git pull --ff-only
else
  echo "Bukan repo git; memakai file yang sudah disalin ke folder ini."
fi

docker compose up -d --build
docker image prune -f >/dev/null

for _ in $(seq 1 30); do
  if curl -fs http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "Statify berjalan."
    docker compose ps
    exit 0
  fi
  sleep 3
done
echo "Statify belum merespons. Cek log: docker compose logs --tail 50" >&2
exit 1
