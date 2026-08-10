#!/usr/bin/env python3
"""
Self-signed TLS for the counter LAN.

Why this exists
---------------
The live screen captures from the camera of whatever device opens it, so a
phone or a laptop on the pharmacy Wi-Fi becomes a scanner. Browsers only give
a page camera access in a *secure context*: HTTPS, or localhost. Plain HTTP on
a LAN address -- exactly how another device reaches this server -- is refused,
with no prompt and no way for the user to override it.

There is no certificate authority on an offline pharmacy network, so the
certificate is generated locally and signed by itself. The browser will warn
once per device that the issuer is unknown; that warning is accurate, and the
operator accepts it deliberately. What it buys is the camera API, plus
encryption of the scan traffic on the local network.

The LAN address is written into subjectAltName. Without it, Safari and Chrome
reject the certificate outright rather than offering the "proceed anyway"
route, because a bare CN has not been accepted for host matching for years.

The key never leaves the machine and is written with owner-only permissions.
Regenerate by deleting the pair; a new one is issued on the next start.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CERT_NAME = "edgemedicheck-cert.pem"
KEY_NAME = "edgemedicheck-key.pem"
VALID_DAYS = 825  # the longest span browsers still accept for a leaf cert


def ensure_certificate(
    directory: Path | str,
    hosts: list[str] | None = None,
) -> tuple[Path, Path]:
    """Return (cert_path, key_path), generating the pair if it is absent."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "HTTPS needs the 'cryptography' package: pip install cryptography"
        ) from exc

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cert_path = directory / CERT_NAME
    key_path = directory / KEY_NAME

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    hosts = hosts or []
    names: list = [x509.DNSName("localhost")]
    seen = {"localhost"}
    for h in hosts:
        if not h or h in seen:
            continue
        seen.add(h)
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            names.append(x509.DNSName(h))
    names.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "EdgeMediCheck scanner"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EdgeMediCheck"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                       critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    os.chmod(key_path, 0o600)

    log.info("Generated self-signed certificate for %s", sorted(seen))
    return cert_path, key_path
