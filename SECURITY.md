# Security Policy

## Supported Versions

Only the latest release receives security fixes. We do not backport security patches to older versions.

| Version | Supported |
|---------|-----------|
| `0.2.x` (latest) | Yes |
| `< 0.2.0` | No |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in skeval, please report it privately using one of the following methods:

- **GitHub Private Advisory:** [Report a vulnerability](https://github.com/skeval-ai/skeval/security/advisories/new) *(preferred)*
- **Email:** Contact the maintainers directly via the email on the GitHub profile

### What to include

Please provide as much of the following as possible:

- A clear description of the vulnerability
- Steps to reproduce the issue
- The potential impact (data exposure, code execution, etc.)
- Your suggested fix (if any)
- Your name/handle for acknowledgement (optional)

### Response timeline

| Stage | Target time |
|---|---|
| Initial acknowledgement | 48 hours |
| Severity assessment | 5 business days |
| Fix and patch release | Depends on severity |

We will notify you when the issue is resolved and credit you in the release notes unless you prefer to remain anonymous.

## Scope

This policy covers the `skeval` Python package published on PyPI and the source code in this repository.

It does not cover:
- Third-party dependencies (report those to the respective maintainers)
- Issues in user-written code that uses skeval

## Security Practices

- Dependencies are scanned automatically on every PR with **pip-audit**
- Source code is scanned on every PR with **bandit**
- Dependency updates are automated via **Dependabot**
- CodeQL analysis runs on every push to `main`
