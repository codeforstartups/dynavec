# Security Policy

## Supported Versions

dynavec is pre-1.0 and under active development. Security fixes are applied to the latest
released version on PyPI. Please make sure you are on the most recent release before
reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub's **[Report a vulnerability](https://github.com/codeforstartups/dynavec/security/advisories/new)**
feature (the *Security* tab → *Report a vulnerability*). This opens a private advisory
visible only to the maintainers.

Please include:

- a description of the vulnerability and its impact,
- steps to reproduce (a minimal proof-of-concept if possible),
- affected version(s) and environment, and
- any suggested remediation.

**What to expect:**

- We aim to acknowledge your report within a few business days.
- We will confirm the issue, keep you updated on progress, and coordinate a fix.
- Once a fix is released, we will credit you in the advisory unless you prefer to remain
  anonymous.

## Scope

dynavec runs entirely inside **your own AWS account** and stores no data outside it. The
most valuable reports concern:

- credential/permission handling (`credentials.py`, IAM guidance),
- injection or escaping issues in stored keys, metadata filters, or graph exports,
- unsafe deserialization or resource-exhaustion paths, and
- anything that could cause data to leak outside the user's account or region.

Thank you for helping keep dynavec and its users safe.
