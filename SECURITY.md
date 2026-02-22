# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please **do not open a public GitHub issue**.
Instead, report it privately by emailing the repository maintainer directly or by using
[GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability).

Please include:

1. A clear description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if available)

You can expect an acknowledgement within **72 hours** and a resolution timeline within **14 days**
for critical issues.

---

## Threat Model

### Assets

| Asset | Sensitivity |
|-------|-------------|
| KuCoin API key / secret / passphrase | **Critical** — full trading access to the account |
| USDT balance / open positions | **High** — financial loss on compromise |
| Bot configuration (trade fraction, SL/TP) | **Medium** — manipulated settings lead to outsized risk |
| Log files | **Low** — may contain symbol names and equity snapshots |

### Trust Boundaries

* The bot runs on a **single host** controlled by the operator.  No web-facing interface is exposed.
* All external communication is outbound-only to the KuCoin REST API over HTTPS.
* No user-supplied input is accepted at runtime.

### Threat Scenarios

| Threat | Likelihood | Mitigation |
|--------|-----------|------------|
| API credentials leaked via `.env` committed to VCS | Medium | `.env` is in `.gitignore`; use `.env.example` as template |
| Compromised dependency supply-chain | Low-Medium | Pin deps in `requirements.txt`; run `pip audit` |
| Bot configured with dangerously high `TRADE_FRACTION` | Medium | Default 0.95 is aggressive; operator should lower for safety |
| Runaway trading loop from unhandled exception | Low | Main loop catches all exceptions and logs them; equity floor halts the bot |
| Hardcoded secrets in source code | Low | Credentials are read exclusively from environment variables; config.py has no defaults |

### Mitigations in Place

* **Credentials** — loaded exclusively from environment variables; never hard-coded.
* **Paper mode** — run with `PAPER_MODE=true` to validate configuration without risking real funds.
* **Equity floor** — `EQUITY_FLOOR_USDT` halts the bot if equity drops below a configurable threshold.
* **Circuit breaker** — `MAX_CONSECUTIVE_LOSSES` pauses trading after a losing streak.
* **HTTPS only** — all API calls go to the official KuCoin SDK which uses HTTPS.

---

## Supported Versions

This project does not yet follow a formal release cycle.  All fixes are applied to the `main` branch.
