# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `.env.example` — documented template for all configurable environment variables.
- `Makefile` — single-command developer experience: `make install`, `make dev`, `make test`, `make lint`.
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline (lint + unit tests on every push/PR).
- `SECURITY.md` — vulnerability reporting guidance and threat model.
- `CHANGELOG.md` — this file, following Keep a Changelog format.

### Fixed
- Resolved 10 `ruff` lint warnings in `tests/test_bot.py` (E402, F841, E741).

### Changed
- README updated with "What's working now" checklist and consolidated setup/run/test commands.
