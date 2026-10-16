#!/usr/bin/env bash
# Setup awal VPS Ubuntu 22.04/24.04 (2 vCPU / 4 GB) untuk Statify.
# Pakai (dari folder proyek):
#   sudo DOMAIN=statify.domainanda.id EMAIL=anda@email.com bash deploy/setup_server.sh
# Opsional: SSH_PORT=2222 jika SSH server Anda tidak memakai port 22.
set -euo pipefail

DOMAIN="${DOMAIN:?Isi DOMAIN, contoh DOMAIN=statify.domainanda.id}"
EMAIL="${EMAIL:?Isi EMAIL untuk sertifikat HTTPS}"
SSH_PORT="${SSH_PORT:-22}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "Jalankan dengan sudo." >&2
  exit 1
fi

echo "==> Swap 2 GB"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q "/swapfile" /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
sysctl -w vm.swappiness=10 >/dev/null
grep -q "vm.swappiness" /etc/sysctl.conf || echo "vm.swappiness=10" >> /etc/sysctl.conf

echo "==> Paket: Docker, Nginx, Certbot, UFW"
apt-get update
apt-get install -y ca-certificates curl nginx certbot python3-certbot-nginx ufw
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi

echo "==> Firewall (SSH port ${SSH_PORT}, HTTP, HTTPS)"
ufw allow "${SSH_PORT}/tcp"
ufw allow "Nginx Full"
ufw --force enable

echo "==> Build & jalankan Statify (build pertama bisa 5-10 menit)"
cd "$APP_DIR"
docker compose up -d --build

echo "==> Menunggu aplikasi siap"
for _ in $(seq 1 30); do
  if curl -fs http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "    aplikasi siap"
    break
  fi
  sleep 3
done

echo "==> Nginx"
sed "s/statify.domainanda.id/${DOMAIN}/" deploy/nginx-statify.conf > /etc/nginx/sites-available/statify
ln -sf /etc/nginx/sites-available/statify /etc/nginx/sites-enabled/statify
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "==> HTTPS"
SERVER_IP="$(curl -fs https://api.ipify.org || true)"
DNS_IP="$(getent ahostsv4 "$DOMAIN" | awk 'NR==1 {print $1}' || true)"
if [[ -n "$SERVER_IP" && "$DNS_IP" != "$SERVER_IP" ]]; then
  echo "PERINGATAN: ${DOMAIN} mengarah ke '${DNS_IP:-tidak ada}', bukan ke server ini (${SERVER_IP})."
  echo "Aplikasi sudah jalan di http://${SERVER_IP}. Setelah DNS benar, jalankan:"
  echo "  sudo certbot --nginx -d ${DOMAIN} -m ${EMAIL} --agree-tos --redirect"
  exit 0
fi
certbot --nginx -d "$DOMAIN" -m "$EMAIL" --agree-tos --non-interactive --redirect

echo "Selesai: https://${DOMAIN}"
