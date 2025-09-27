#!/usr/bin/env bash
set -euo pipefail

# Provision/renew Let's Encrypt certificates for api.example.com using certbot + Nginx.
# Usage: sudo ./generate_tls.sh api.example.com you@example.com

DOMAIN=${1:-api.example.com}
EMAIL=${2:-admin@example.com}

if ! command -v certbot >/dev/null 2>&1; then
  echo "Installing certbot..."
  sudo apt-get update
  sudo apt-get install -y certbot python3-certbot-nginx
fi

sudo mkdir -p /var/www/certbot

sudo certbot certonly \
  --nginx \
  --agree-tos \
  --non-interactive \
  --email "${EMAIL}" \
  --domain "${DOMAIN}" \
  --deploy-hook "systemctl reload nginx"

echo "Certificates installed under /etc/letsencrypt/live/${DOMAIN}."
echo "Add a cron entry: 0 3 * * * root certbot renew --quiet"
