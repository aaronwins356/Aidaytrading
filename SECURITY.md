# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` branch | ✅ |
| Tagged releases | ✅ |

Older snapshots are unsupported; upgrade to the latest tag or `main` before requesting assistance.

## Reporting a Vulnerability

- **Email:** security@aidaytrading.example
- **GPG:** https://aidaytrading.example/pgp/security.asc
- **Response SLA:** We acknowledge reports within 2 business days and aim to provide a remediation plan within 7 business days.

### Scope

- `backend/` FastAPI service (authentication, admin, health/metrics).
- `ai_trader/` runtime and its FastAPI/Streamlit endpoints when deployed under the Aidaytrading brand.
- Infrastructure guidance contained in `deploy/` and associated automation.

### Out of Scope

- Third-party services (Brevo SMTP, Kraken APIs) unless a misconfiguration in our code triggers the issue.
- Social engineering, phishing, or denial-of-service attacks without a reproducible software flaw.
- Vulnerabilities in forks or downstream modifications.

### Safe Harbour

We will not pursue legal action for good-faith, non-destructive testing performed within scope. Avoid accessing real customer data; use demo accounts and local environments where possible.

Include detailed reproduction steps, affected endpoints, logs (with secrets redacted), and suggested fixes if known. Encrypt sensitive reports with our public key.
