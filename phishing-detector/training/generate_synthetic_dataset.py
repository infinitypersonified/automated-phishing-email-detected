from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


PHISHING_SUBJECTS = [
    "Account suspended",
    "Urgent payment verification",
    "Security alert",
    "Payroll correction required",
    "Invoice overdue notice",
    "Mailbox quota exceeded",
    "Unusual sign-in activity",
    "Benefits update required",
]

PHISHING_BODIES = [
    "Your account has been suspended. Verify immediately at {url}.",
    "We detected unauthorized login. Confirm identity using {url}.",
    "Final warning: update payment details now via {url}.",
    "Your payroll information is invalid. Resubmit credentials at {url}.",
    "A secure document is waiting. Open and verify at {url}.",
    "To avoid service interruption, validate your mailbox at {url}.",
]

LEGIT_SUBJECTS = [
    "Weekly team update",
    "Project status meeting",
    "Training schedule",
    "Quarterly planning notes",
    "Code review reminder",
    "Client feedback summary",
    "Office maintenance notice",
    "Release checklist",
]

LEGIT_BODIES = [
    "Hi team, please review the attached project update before tomorrow's standup.",
    "Reminder: our planning session is scheduled for 2 PM in the main conference room.",
    "Please find the weekly report attached. Let me know if any section needs updates.",
    "The onboarding workshop starts Monday. Agenda and materials are in the shared drive.",
    "Thanks for your input on the feature draft. We will discuss next steps in sprint planning.",
    "Please approve the pull requests assigned to you by end of day.",
]

DOMAINS = [
    "secure-check-alert.com",
    "verify-account-now.net",
    "auth-confirm-center.org",
    "mail-protect-signin.com",
    "billing-update-portal.info",
]


def phishing_row() -> tuple[str, str]:
    subject = random.choice(PHISHING_SUBJECTS)
    sender = random.choice(
        [
            "security@apple-security-team.com",
            "noreply@microsoft-alerts.net",
            "admin@paypal-check-center.org",
            "support@banking-update-secure.com",
        ]
    )
    ip_fragment = ".".join(str(random.randint(2, 250)) for _ in range(4))
    domain = random.choice(DOMAINS)
    url = random.choice(
        [
            f"http://{ip_fragment}/verify",
            f"https://{domain}/login",
            f"https://{domain}/validate/account",
        ]
    )
    urgency = random.choice(
        [
            "This link expires in 24 hours.",
            "Immediate action is required.",
            "Failure to act will result in account closure.",
            "Respond within 72 hours.",
        ]
    )
    body = random.choice(PHISHING_BODIES).format(url=url)
    text = f"From: {sender}\nSubject: {subject}\n{body} {urgency}"
    return text, "phishing"


def legitimate_row() -> tuple[str, str]:
    subject = random.choice(LEGIT_SUBJECTS)
    sender = random.choice(
        [
            "manager@company.com",
            "hr@company.com",
            "teammate@company.com",
            "it-support@company.com",
        ]
    )
    body = random.choice(LEGIT_BODIES)
    text = f"From: {sender}\nSubject: {subject}\n{body}"
    return text, "legitimate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--phishing", type=int, default=5000, help="Phishing rows")
    parser.add_argument("--legitimate", type=int, default=5000, help="Legitimate rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows: list[tuple[str, str]] = []
    rows.extend(phishing_row() for _ in range(args.phishing))
    rows.extend(legitimate_row() for _ in range(args.legitimate))
    random.shuffle(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
