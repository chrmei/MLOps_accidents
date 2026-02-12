# GO-LIVE Checklist

Before production deployment, complete these items:

## Secrets and Config

- [ ] Set strong `JWT_SECRET_KEY` (never use default `CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET_KEY`)
- [ ] Change default admin password (`ADMIN_PASSWORD`); do not use default in production

## Security

- [ ] Set `CORS_ORIGINS` to actual front-end origins (no wildcards in production)
- [ ] Review rate limits and lockout settings (per-username limits, lockout duration in `config.py`)

## Infrastructure

- [ ] Enable HTTPS when implemented
- [ ] Replace in-memory user store with persistent database for production
- [ ] For multi-instance: replace in-memory rate/lockout/blocklist with a shared store (e.g. Redis)

## Application

- [ ] Token revocation: in-memory blocklist is single-instance only; for multi-instance use a shared store (e.g. Redis)
- [ ] Ensure auth events (failed login, lockout) are logged for audit (logging is configured in auth routes)
