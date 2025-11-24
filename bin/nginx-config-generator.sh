#!/bin/bash

##############################################################################
# Universal Nginx Site Configuration Generator
#
# Features:
# - Automatic site type detection (static, PHP, Node.js, reverse proxy)
# - Certbot SSL certificates via Cloudflare DNS
# - Optimized for speed, security, and caching
# - Cloudflare proxy ready (settings commented out)
# - Comprehensive error handling and rollback
##############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}==>${NC} ${BOLD}$1${NC}\n"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root (use sudo)"
   exit 1
fi

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="/root/nginx-backup-$(date +%Y%m%d-%H%M%S)"
CLOUDFLARE_INI="/etc/letsencrypt/cloudflare.ini"

##############################################################################
# Banner
##############################################################################

clear
echo "============================================================================"
echo "           Universal Nginx Site Configuration Generator"
echo "============================================================================"
echo ""
echo "This script will:"
echo "  1. Detect your site type (static, PHP, Node.js, or reverse proxy)"
echo "  2. Generate SSL certificates via Certbot (Cloudflare DNS)"
echo "  3. Create optimized nginx configuration"
echo "  4. Configure for direct access (Cloudflare settings commented out)"
echo ""
echo "============================================================================"
echo ""

##############################################################################
# Check prerequisites
##############################################################################

log_step "Checking prerequisites..."

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    log_error "nginx is not installed"
    echo "Install with: apt-get install nginx"
    exit 1
fi

# Check if certbot is installed
if ! command -v certbot &> /dev/null; then
    log_warning "certbot is not installed, installing..."
    apt-get update -qq
    apt-get install -y certbot python3-certbot-nginx
fi

# Check if Cloudflare DNS plugin is installed
if ! dpkg -l | grep -q python3-certbot-dns-cloudflare; then
    log_warning "Cloudflare DNS plugin not installed, installing..."
    apt-get install -y python3-certbot-dns-cloudflare
fi

# Check if Cloudflare credentials exist
if [[ ! -f "$CLOUDFLARE_INI" ]]; then
    log_error "Cloudflare credentials file not found: $CLOUDFLARE_INI"
    echo ""
    echo "Create this file with your Cloudflare API token:"
    echo ""
    echo "sudo mkdir -p /etc/letsencrypt"
    echo "sudo tee $CLOUDFLARE_INI > /dev/null <<EOF"
    echo "# Cloudflare API token"
    echo "dns_cloudflare_api_token = YOUR_API_TOKEN_HERE"
    echo "EOF"
    echo "sudo chmod 600 $CLOUDFLARE_INI"
    echo ""
    echo "Get your API token from: https://dash.cloudflare.com/profile/api-tokens"
    echo "Token needs: Zone:DNS:Edit permissions"
    exit 1
fi

# Verify Cloudflare credentials file permissions
PERMS=$(stat -c %a "$CLOUDFLARE_INI")
if [[ "$PERMS" != "600" ]]; then
    log_warning "Fixing Cloudflare credentials file permissions..."
    chmod 600 "$CLOUDFLARE_INI"
fi

log_success "All prerequisites met"

##############################################################################
# Collect user input
##############################################################################

log_step "Collecting site information..."

# Domain name
while true; do
    read -p "Enter domain name (e.g., example.com): " DOMAIN
    DOMAIN=$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | sed 's/^www\.//')

    if [[ -z "$DOMAIN" ]]; then
        log_error "Domain cannot be empty"
        continue
    fi

    if [[ ! "$DOMAIN" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$ ]]; then
        log_error "Invalid domain format"
        continue
    fi

    break
done

WWW_DOMAIN="www.$DOMAIN"
log_info "Domain: $DOMAIN"
log_info "WWW Domain: $WWW_DOMAIN"

# Subdomain handling
read -p "Is this a subdomain? (y/n, default: n): " IS_SUBDOMAIN
IS_SUBDOMAIN=${IS_SUBDOMAIN:-n}

if [[ "$IS_SUBDOMAIN" =~ ^[Yy]$ ]]; then
    USE_WWW="n"
    PRIMARY_DOMAIN="$DOMAIN"
    log_info "Subdomain mode: Will not create www redirect"
else
    # Ask about www preference
    read -p "Use www prefix as primary? (y/n, default: y): " USE_WWW
    USE_WWW=${USE_WWW:-y}

    if [[ "$USE_WWW" =~ ^[Yy]$ ]]; then
        PRIMARY_DOMAIN="$WWW_DOMAIN"
        REDIRECT_DOMAIN="$DOMAIN"
    else
        PRIMARY_DOMAIN="$DOMAIN"
        REDIRECT_DOMAIN="$WWW_DOMAIN"
    fi
    log_info "Primary domain: $PRIMARY_DOMAIN"
fi

# Site type
echo ""
echo "Site type options:"
echo "  1) Static site (HTML/CSS/JS files)"
echo "  2) PHP site (requires PHP-FPM)"
echo "  3) Node.js / Python / Backend application (reverse proxy)"
echo "  4) Auto-detect (scan directory for site type)"
echo ""
read -p "Select site type (1-4, default: 4): " SITE_TYPE_CHOICE
SITE_TYPE_CHOICE=${SITE_TYPE_CHOICE:-4}

# Get site location or backend URL
if [[ "$SITE_TYPE_CHOICE" == "3" ]]; then
    read -p "Enter backend URL (e.g., http://localhost:3000): " BACKEND_URL

    if [[ ! "$BACKEND_URL" =~ ^https?:// ]]; then
        log_error "Backend URL must start with http:// or https://"
        exit 1
    fi

    SITE_ROOT=""
    SITE_TYPE="proxy"
    log_info "Backend URL: $BACKEND_URL"
else
    read -p "Enter site root directory (e.g., /var/www/example.com): " SITE_ROOT

    if [[ -z "$SITE_ROOT" ]]; then
        log_error "Site root cannot be empty"
        exit 1
    fi

    # Create directory if it doesn't exist
    if [[ ! -d "$SITE_ROOT" ]]; then
        log_warning "Directory doesn't exist, creating: $SITE_ROOT"
        mkdir -p "$SITE_ROOT"
        chown -R www-data:www-data "$SITE_ROOT"
    fi

    # Detect site type if auto-detect chosen
    if [[ "$SITE_TYPE_CHOICE" == "4" ]]; then
        log_info "Auto-detecting site type..."

        if find "$SITE_ROOT" -maxdepth 2 -name "*.php" -type f | grep -q .; then
            SITE_TYPE="php"
            log_success "Detected: PHP site"
        elif [[ -f "$SITE_ROOT/package.json" ]] || [[ -f "$SITE_ROOT/../package.json" ]]; then
            SITE_TYPE="nodejs"
            log_success "Detected: Node.js site"
        else
            SITE_TYPE="static"
            log_success "Detected: Static site"
        fi
    else
        case "$SITE_TYPE_CHOICE" in
            1) SITE_TYPE="static" ;;
            2) SITE_TYPE="php" ;;
            *) SITE_TYPE="static" ;;
        esac
    fi

    log_info "Site type: $SITE_TYPE"
    log_info "Site root: $SITE_ROOT"
fi

# Confirm before proceeding
echo ""
echo "============================================================================"
echo "Configuration Summary:"
echo "============================================================================"
echo "Domain: $PRIMARY_DOMAIN"
if [[ "$IS_SUBDOMAIN" =~ ^[Nn]$ ]]; then
    echo "Redirect from: $REDIRECT_DOMAIN"
fi
echo "Site type: $SITE_TYPE"
if [[ "$SITE_TYPE" == "proxy" ]]; then
    echo "Backend: $BACKEND_URL"
else
    echo "Root directory: $SITE_ROOT"
fi
echo "============================================================================"
echo ""
read -p "Proceed with this configuration? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    log_error "Configuration cancelled by user"
    exit 1
fi

##############################################################################
# Create backup
##############################################################################

log_step "Creating backup..."
mkdir -p "$BACKUP_DIR"
if [[ -f "/etc/nginx/sites-available/$DOMAIN.conf" ]]; then
    cp "/etc/nginx/sites-available/$DOMAIN.conf" "$BACKUP_DIR/$DOMAIN.conf"
    log_success "Backed up existing configuration"
fi

##############################################################################
# Check/Install PHP-FPM if needed
##############################################################################

if [[ "$SITE_TYPE" == "php" ]]; then
    log_step "Checking PHP-FPM installation..."

    if ! command -v php-fpm &> /dev/null && ! systemctl list-units --full -all | grep -q php.*fpm; then
        log_warning "PHP-FPM not found, installing..."
        apt-get update -qq
        apt-get install -y php-fpm php-mysql php-xml php-mbstring php-curl php-zip php-gd
    fi

    # Detect PHP-FPM version and socket
    PHP_VERSION=$(php -r 'echo PHP_MAJOR_VERSION.".".PHP_MINOR_VERSION;' 2>/dev/null || echo "8.3")
    PHP_SOCKET="/run/php/php${PHP_VERSION}-fpm.sock"

    if [[ ! -S "$PHP_SOCKET" ]]; then
        # Try to find any PHP-FPM socket
        PHP_SOCKET=$(find /run/php -name "php*-fpm.sock" -type s | head -1)
        if [[ -z "$PHP_SOCKET" ]]; then
            log_error "Could not find PHP-FPM socket"
            exit 1
        fi
    fi

    log_success "PHP-FPM found: $PHP_SOCKET"
fi

##############################################################################
# Generate SSL certificates
##############################################################################

log_step "Generating SSL certificates with Cloudflare DNS..."

# Determine which domains to certify
if [[ "$IS_SUBDOMAIN" =~ ^[Yy]$ ]]; then
    CERT_DOMAINS="-d $DOMAIN"
    CERT_NAME="$DOMAIN"
else
    CERT_DOMAINS="-d $DOMAIN -d $WWW_DOMAIN"
    CERT_NAME="$DOMAIN"
fi

# Check if certificates already exist
if [[ -d "/etc/letsencrypt/live/$CERT_NAME" ]]; then
    log_warning "Certificates already exist for $CERT_NAME"
    read -p "Renew certificates? (y/n, default: n): " RENEW_CERT
    RENEW_CERT=${RENEW_CERT:-n}

    if [[ "$RENEW_CERT" =~ ^[Yy]$ ]]; then
        log_info "Renewing certificates..."
        certbot certonly \
            --dns-cloudflare \
            --dns-cloudflare-credentials "$CLOUDFLARE_INI" \
            --dns-cloudflare-propagation-seconds 30 \
            $CERT_DOMAINS \
            --force-renewal \
            --non-interactive \
            --agree-tos \
            --email admin@$DOMAIN

        if [[ $? -eq 0 ]]; then
            log_success "Certificates renewed successfully"
        else
            log_error "Certificate renewal failed"
            exit 1
        fi
    else
        log_info "Using existing certificates"
    fi
else
    log_info "Requesting new certificates..."
    certbot certonly \
        --dns-cloudflare \
        --dns-cloudflare-credentials "$CLOUDFLARE_INI" \
        --dns-cloudflare-propagation-seconds 30 \
        $CERT_DOMAINS \
        --non-interactive \
        --agree-tos \
        --email admin@$DOMAIN

    if [[ $? -eq 0 ]]; then
        log_success "Certificates obtained successfully"
    else
        log_error "Certificate generation failed"
        log_info "Check your Cloudflare API token and DNS settings"
        exit 1
    fi
fi

##############################################################################
# Generate nginx configuration
##############################################################################

log_step "Generating nginx configuration..."

CONFIG_FILE="/etc/nginx/sites-available/$DOMAIN.conf"

# Create base configuration header
cat > "$CONFIG_FILE" <<EOF
# Nginx configuration for $DOMAIN
# Generated by Universal Site Generator on $(date)
# Site type: $SITE_TYPE
# Optimized for speed, security, and caching
# Cloudflare proxy settings are commented out but ready for use

EOF

# Add redirect server blocks (if not a subdomain)
if [[ "$IS_SUBDOMAIN" =~ ^[Nn]$ ]]; then
    cat >> "$CONFIG_FILE" <<EOF
##
# Redirect non-primary HTTP to primary HTTPS
##
server {
    listen 80;
    listen [::]:80;
    server_name $REDIRECT_DOMAIN;

    # Security headers even on redirects
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Redirect to primary with HTTPS
    return 301 https://$PRIMARY_DOMAIN\$request_uri;
}

##
# Redirect non-primary HTTPS to primary HTTPS
##
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $REDIRECT_DOMAIN;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/$CERT_NAME/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$CERT_NAME/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/$CERT_NAME/chain.pem;

    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Redirect to primary
    return 301 https://$PRIMARY_DOMAIN\$request_uri;
}

EOF
fi

# Add HTTP to HTTPS redirect for primary domain
cat >> "$CONFIG_FILE" <<EOF
##
# Redirect primary HTTP to primary HTTPS
##
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name $PRIMARY_DOMAIN;

    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Allow Let's Encrypt verification
    location ^~ /.well-known/acme-challenge/ {
        default_type "text/plain";
        root /var/www/html;
        allow all;
    }

    # Redirect everything else to HTTPS
    location / {
        return 301 https://$PRIMARY_DOMAIN\$request_uri;
    }
}

##
# Main HTTPS server block for $PRIMARY_DOMAIN
##
server {
    listen 443 ssl http2 default_server;
    listen [::]:443 ssl http2 default_server;
    server_name $PRIMARY_DOMAIN;

EOF

# Add site root or proxy configuration
if [[ "$SITE_TYPE" == "proxy" ]]; then
    cat >> "$CONFIG_FILE" <<EOF
    # Reverse proxy configuration
    # Backend: $BACKEND_URL

EOF
else
    cat >> "$CONFIG_FILE" <<EOF
    # Document root
    root $SITE_ROOT;
    index index.html index.htm$(if [[ "$SITE_TYPE" == "php" ]]; then echo " index.php"; fi);

    # Character set
    charset utf-8;

EOF
fi

# SSL configuration
cat >> "$CONFIG_FILE" <<EOF
    ##
    # SSL Configuration
    ##

    ssl_certificate /etc/letsencrypt/live/$CERT_NAME/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$CERT_NAME/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/$CERT_NAME/chain.pem;

    # DH parameters for perfect forward secrecy
    ssl_dhparam /etc/ssl/dhparams.pem;

    # Use modern SSL settings from main nginx.conf
    include /etc/letsencrypt/options-ssl-nginx.conf;

    ##
    # Cloudflare Settings (COMMENTED OUT - for direct access)
    ##

    # Uncomment when using Cloudflare proxy:
    # include /etc/nginx/snippets/cloudflare-ips.conf;

    # Cloudflare Authenticated Origin Pulls
    # Uncomment after enabling in Cloudflare dashboard:
    # ssl_client_certificate /etc/ssl/certs/cloudflare.pem;
    # ssl_verify_client on;

    ##
    # Logging
    ##

    access_log /var/log/nginx/$PRIMARY_DOMAIN.access.log combined;
    error_log /var/log/nginx/$PRIMARY_DOMAIN.error.log warn;

    ##
    # Security Headers
    ##

    # HSTS with preload (forces HTTPS)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Prevent MIME type sniffing
    add_header X-Content-Type-Options "nosniff" always;

    # XSS Protection (legacy but still useful)
    add_header X-XSS-Protection "1; mode=block" always;

    # Referrer policy
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Prevent embedding in frames
    add_header X-Frame-Options "DENY" always;

    # Control Flash/PDF policies
    add_header X-Permitted-Cross-Domain-Policies "none" always;

    ##
    # Rate Limiting
    ##

    # Apply general rate limit
    limit_req zone=general burst=20 nodelay;

    ##
    # Performance: Gzip (already enabled in main config)
    ##

    gzip_vary on;

    ##
    # Let's Encrypt renewal
    ##

    location ^~ /.well-known/acme-challenge/ {
        default_type "text/plain";
        root /var/www/html;
        allow all;
    }

    ##
    # Deny access to hidden files
    ##

    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }

EOF

# Add site-type specific configuration
case "$SITE_TYPE" in
    "static")
        cat >> "$CONFIG_FILE" <<'EOF'
    ##
    # Favicon and robots.txt
    ##

    location = /favicon.ico {
        log_not_found off;
        access_log off;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location = /robots.txt {
        allow all;
        log_not_found off;
        access_log off;
    }

    ##
    # Static asset caching with optimal headers
    ##

    # Images
    location ~* \.(?:jpg|jpeg|gif|png|ico|webp|avif|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;

        add_header X-Content-Type-Options "nosniff" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    # CSS and JavaScript
    location ~* \.(?:css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;

        add_header X-Content-Type-Options "nosniff" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    # Fonts
    location ~* \.(?:woff|woff2|ttf|otf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Access-Control-Allow-Origin "*";
        access_log off;

        add_header X-Content-Type-Options "nosniff" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    # Media files
    location ~* \.(?:mp4|webm|ogg|mp3|wav|flac|aac|opus)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;

        add_header X-Content-Type-Options "nosniff" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    ##
    # Main location block
    ##

    location / {
        try_files $uri $uri/ =404;

        # HTML caching (shorter than static assets)
        location ~* \.html$ {
            expires 1h;
            add_header Cache-Control "public, must-revalidate";

            add_header X-Content-Type-Options "nosniff" always;
            add_header X-XSS-Protection "1; mode=block" always;
            add_header X-Frame-Options "DENY" always;
            add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
            add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        }
    }
EOF
        ;;

    "php")
        cat >> "$CONFIG_FILE" <<EOF
    ##
    # Block common exploit attempts
    ##

    location ~* /(?:uploads|files|wp-content|wp-includes)/.*.php$ {
        deny all;
        access_log off;
        log_not_found off;
    }

    ##
    # Favicon and robots.txt
    ##

    location = /favicon.ico {
        log_not_found off;
        access_log off;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location = /robots.txt {
        allow all;
        log_not_found off;
        access_log off;
    }

    ##
    # Static asset caching
    ##

    location ~* \.(?:jpg|jpeg|gif|png|ico|webp|avif|svg|css|js|woff|woff2|ttf|otf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;

        add_header X-Content-Type-Options "nosniff" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    ##
    # PHP-FPM configuration
    ##

    location ~ \.php$ {
        try_files \$uri =404;
        fastcgi_split_path_info ^(.+\.php)(/.+)$;
        fastcgi_pass unix:$PHP_SOCKET;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME \$document_root\$fastcgi_script_name;
        include fastcgi_params;

        # PHP-FPM optimizations
        fastcgi_buffers 16 32k;
        fastcgi_buffer_size 64k;
        fastcgi_busy_buffers_size 256k;
        fastcgi_connect_timeout 300;
        fastcgi_send_timeout 300;
        fastcgi_read_timeout 300;

        # Security
        fastcgi_param PHP_VALUE "open_basedir=$SITE_ROOT:/tmp";
        fastcgi_param PHP_ADMIN_VALUE "disable_functions=exec,passthru,shell_exec,system";
    }

    ##
    # Main location block
    ##

    location / {
        try_files \$uri \$uri/ /index.php?\$args;
    }
EOF
        ;;

    "proxy")
        cat >> "$CONFIG_FILE" <<EOF
    ##
    # Reverse Proxy Configuration
    ##

    location / {
        proxy_pass $BACKEND_URL;
        proxy_http_version 1.1;

        # Pass original request info
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Port \$server_port;

        # WebSocket support
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;

        # Don't pass certain headers to backend
        proxy_hide_header X-Powered-By;

        # Add security headers
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    }

    ##
    # Static assets (if your backend serves them)
    ##

    location ~* \.(?:jpg|jpeg|gif|png|ico|webp|avif|svg|css|js|woff|woff2|ttf|otf|eot)$ {
        proxy_pass $BACKEND_URL;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;

        # Cache these on nginx side
        proxy_cache_valid 200 1y;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
EOF
        ;;
esac

# Close server block
cat >> "$CONFIG_FILE" <<'EOF'
}
EOF

log_success "Configuration file created: $CONFIG_FILE"

##############################################################################
# Enable site
##############################################################################

log_step "Enabling site..."

# Create symlink
if [[ -L "/etc/nginx/sites-enabled/$DOMAIN.conf" ]]; then
    log_info "Site already enabled"
else
    ln -sf "/etc/nginx/sites-available/$DOMAIN.conf" "/etc/nginx/sites-enabled/$DOMAIN.conf"
    log_success "Site enabled"
fi

##############################################################################
# Create test page if directory is empty
##############################################################################

if [[ "$SITE_TYPE" != "proxy" ]] && [[ -d "$SITE_ROOT" ]]; then
    if ! find "$SITE_ROOT" -mindepth 1 -print -quit | grep -q .; then
        log_info "Creating test page..."

        if [[ "$SITE_TYPE" == "php" ]]; then
            cat > "$SITE_ROOT/index.php" <<EOF
<?php
phpinfo();
EOF
            chown www-data:www-data "$SITE_ROOT/index.php"
            log_success "Created PHP test page"
        else
            cat > "$SITE_ROOT/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$PRIMARY_DOMAIN</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        h1 {
            font-size: 3rem;
            margin: 0;
            margin-bottom: 1rem;
        }
        p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>$PRIMARY_DOMAIN</h1>
        <p>Site is live and configured!</p>
        <p><small>Nginx + HTTP/2 + SSL</small></p>
    </div>
</body>
</html>
EOF
            chown www-data:www-data "$SITE_ROOT/index.html"
            log_success "Created HTML test page"
        fi
    fi
fi

##############################################################################
# Test nginx configuration
##############################################################################

log_step "Testing nginx configuration..."

if nginx -t 2>&1 | tee /tmp/nginx-test.log; then
    log_success "Nginx configuration test passed!"
else
    log_error "Nginx configuration test failed!"
    cat /tmp/nginx-test.log
    log_error "Rolling back..."

    rm -f "/etc/nginx/sites-enabled/$DOMAIN.conf"
    rm -f "/etc/nginx/sites-available/$DOMAIN.conf"

    if [[ -f "$BACKUP_DIR/$DOMAIN.conf" ]]; then
        cp "$BACKUP_DIR/$DOMAIN.conf" "/etc/nginx/sites-available/$DOMAIN.conf"
        ln -sf "/etc/nginx/sites-available/$DOMAIN.conf" "/etc/nginx/sites-enabled/$DOMAIN.conf"
        log_info "Previous configuration restored"
    fi

    exit 1
fi

##############################################################################
# Reload nginx
##############################################################################

log_step "Reloading nginx..."

if systemctl reload nginx; then
    log_success "Nginx reloaded successfully!"
else
    log_error "Failed to reload nginx"
    log_info "Attempting restart..."
    systemctl restart nginx
fi

##############################################################################
# Verification
##############################################################################

log_step "Running verification tests..."

sleep 2

# Test HTTPS
log_info "Testing HTTPS connection..."
if curl -s -o /dev/null -w "%{http_code}" "https://$PRIMARY_DOMAIN" | grep -q "200\|301\|302"; then
    log_success "HTTPS responding correctly"
else
    log_warning "HTTPS may not be responding (check firewall and DNS)"
fi

# Test HTTP redirect
log_info "Testing HTTP to HTTPS redirect..."
if curl -s -o /dev/null -w "%{http_code}" "http://$PRIMARY_DOMAIN" | grep -q "301"; then
    log_success "HTTP redirects to HTTPS"
else
    log_warning "HTTP redirect may not be working"
fi

##############################################################################
# Summary
##############################################################################

echo ""
echo "============================================================================"
log_success "Site configuration completed successfully!"
echo "============================================================================"
echo ""
echo "Site details:"
echo "  Primary domain: $PRIMARY_DOMAIN"
if [[ "$IS_SUBDOMAIN" =~ ^[Nn]$ ]]; then
    echo "  Redirect domain: $REDIRECT_DOMAIN"
fi
echo "  Site type: $SITE_TYPE"
if [[ "$SITE_TYPE" == "proxy" ]]; then
    echo "  Backend: $BACKEND_URL"
else
    echo "  Root directory: $SITE_ROOT"
fi
echo ""
echo "Files created:"
echo "  Configuration: /etc/nginx/sites-available/$DOMAIN.conf"
echo "  SSL certificate: /etc/letsencrypt/live/$CERT_NAME/"
echo "  Backup: $BACKUP_DIR"
echo ""
echo "Features enabled:"
echo "  ✓ HTTP/2"
echo "  ✓ SSL/TLS (Let's Encrypt)"
echo "  ✓ HTTPS redirects"
echo "  ✓ Security headers (HSTS, CSP, etc.)"
echo "  ✓ Rate limiting"
echo "  ✓ Gzip compression"
if [[ "$SITE_TYPE" == "php" ]]; then
    echo "  ✓ PHP-FPM ($PHP_SOCKET)"
fi
if [[ "$SITE_TYPE" == "proxy" ]]; then
    echo "  ✓ Reverse proxy with WebSocket support"
fi
echo ""
echo "Testing commands:"
echo "  curl -I https://$PRIMARY_DOMAIN"
echo "  curl -I http://$PRIMARY_DOMAIN"
if [[ "$IS_SUBDOMAIN" =~ ^[Nn]$ ]]; then
    echo "  curl -I https://$REDIRECT_DOMAIN"
fi
echo ""
echo "View logs:"
echo "  tail -f /var/log/nginx/$PRIMARY_DOMAIN.access.log"
echo "  tail -f /var/log/nginx/$PRIMARY_DOMAIN.error.log"
echo ""
echo "To enable Cloudflare proxy:"
echo "  1. Set Cloudflare SSL mode to 'Full (strict)'"
echo "  2. Enable 'Authenticated Origin Pulls'"
echo "  3. Set DNS to proxied (orange cloud)"
echo "  4. Uncomment Cloudflare lines in: /etc/nginx/sites-available/$DOMAIN.conf"
echo "  5. Reload nginx: systemctl reload nginx"
echo ""
echo "SSL certificate renewal:"
echo "  Certbot will auto-renew via systemd timer"
echo "  Manual renewal: certbot renew"
echo "  Test renewal: certbot renew --dry-run"
echo ""
echo "============================================================================"
