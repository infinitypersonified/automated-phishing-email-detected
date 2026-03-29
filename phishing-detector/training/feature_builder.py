import re
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "account suspended",
    "password reset",
    "click here",
    "bank",
    "invoice",
    "security alert",
    "confirm",
]

URL_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)
IP_URL_REGEX = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s]*)?", re.IGNORECASE
)
SENDER_REGEX = re.compile(r"from:\s*([^\n]+)", re.IGNORECASE)


def _links(text):
    return URL_REGEX.findall(text or "")


def _domains(links):
    out = set()
    for link in links:
        url = link if link.startswith(("http://", "https://")) else f"http://{link}"
        parsed = urlparse(url)
        if parsed.netloc:
            out.add(parsed.netloc.lower())
    return sorted(out)


def _sender_mismatch(text, domains):
    sender_match = SENDER_REGEX.search(text or "")
    if not sender_match or not domains:
        return 0
    sender_line = sender_match.group(1).lower()
    sender_domains = set(re.findall(r"@([a-z0-9.-]+\.[a-z]{2,})", sender_line))
    if not sender_domains:
        return 0
    return int(sender_domains.isdisjoint(set(domains)))


def _suspicious_count(text):
    t = (text or "").lower()
    return sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in t)


def build_features_dataframe(texts):
    rows = []
    for text in texts:
        links = _links(text)
        domains = _domains(links)
        url_lengths = [len(url) for url in links] or [0]
        rows.append(
            {
                "num_links": len(links),
                "num_domains": len(domains),
                "num_suspicious_words": _suspicious_count(text),
                "url_avg_length": float(sum(url_lengths) / len(url_lengths)),
                "has_ip_url": int(bool(IP_URL_REGEX.search(text or ""))),
                "sender_mismatch": _sender_mismatch(text, domains),
            }
        )
    return pd.DataFrame(rows)
