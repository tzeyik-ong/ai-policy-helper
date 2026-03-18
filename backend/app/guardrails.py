"""PDPA / PII guardrail — redact sensitive identifiers before sending queries to the LLM."""
import re

# Malaysian IC number: YYMMDD-SS-NNNN
_IC = re.compile(r"\b\d{6}-\d{2}-\d{4}\b")

# Malaysian phone: +60 or 01x / 03-xxxx or 011-xxxx etc.
_PHONE = re.compile(r"(\+?60[-\s]?|0)[1-9][\d\s\-]{6,10}\d")

# Generic email
_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def mask_pii(text: str) -> str:
    """Redact IC numbers, phone numbers, and email addresses from user input."""
    text = _IC.sub("[IC_REDACTED]", text)
    text = _PHONE.sub("[PHONE_REDACTED]", text)
    text = _EMAIL.sub("[EMAIL_REDACTED]", text)
    return text
