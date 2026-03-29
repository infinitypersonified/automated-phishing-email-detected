import re
from urllib.parse import urlparse

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


def extract_links(text: str) -> list[str]:
    return URL_REGEX.findall(text or "")


def extract_domains(links: list[str]) -> list[str]:
    domains = set()
    for link in links:
        url = link if link.startswith(("http://", "https://")) else f"http://{link}"
        parsed = urlparse(url)
        if parsed.netloc:
            domains.add(parsed.netloc.lower())
    return sorted(domains)


def detect_suspicious_words(text: str) -> list[str]:
    content = (text or "").lower()
    found = [word for word in SUSPICIOUS_KEYWORDS if word in content]
    return sorted(set(found))


def detect_sender_mismatch(text: str, domains: list[str]) -> int:
    sender_match = SENDER_REGEX.search(text or "")
    if not sender_match or not domains:
        return 0
    sender_line = sender_match.group(1).lower()
    sender_domains = set(re.findall(r"@([a-z0-9.-]+\.[a-z]{2,})", sender_line))
    if not sender_domains:
        return 0
    return int(sender_domains.isdisjoint(set(domains)))


def compute_structural_features(text: str) -> dict:
    links = extract_links(text)
    domains = extract_domains(links)
    suspicious_words = detect_suspicious_words(text)
    url_lengths = [len(url) for url in links] or [0]

    return {
        "num_links": len(links),
        "num_domains": len(domains),
        "num_suspicious_words": len(suspicious_words),
        "url_avg_length": float(sum(url_lengths) / len(url_lengths)),
        "has_ip_url": int(bool(IP_URL_REGEX.search(text or ""))),
        "sender_mismatch": detect_sender_mismatch(text, domains),
        "suspicious_words": suspicious_words,
        "links_detected": links,
        "domains": domains,
    }
