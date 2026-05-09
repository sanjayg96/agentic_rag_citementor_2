import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

def check_input_safety(query: str) -> dict:
    """
    Lightweight input guardrail to detect PII or prompt injection.
    Returns a dict with 'is_safe' boolean and an optional 'reason'.
    """
    # Pattern check for obvious PII (e.g., SSN, Credit Cards, standard phone formats)
    pii_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]*?){13,16}\b')
    if pii_pattern.search(query):
        logger.warning("Input Guardrail triggered: Potential PII detected.")
        return {"is_safe": False, "reason": "Query contains sensitive personal information (PII)."}
        
    # Pattern check for basic Prompt Injection
    injection_keywords = ["ignore previous", "system prompt", "bypass", "jailbreak", "disregard instructions"]
    if any(kw in query.lower() for kw in injection_keywords):
        logger.warning("Input Guardrail triggered: Potential prompt injection.")
        return {"is_safe": False, "reason": "Query violates safety and boundary policies."}
        
    return {"is_safe": True, "reason": ""}

def check_output_safety(answer: str, retrieved_chunks: list[dict[str, Any]] | None = None) -> dict:
    """
    Lightweight output guardrail for the generated answer.
    Blocks obvious policy leaks, sensitive data, and ungrounded empty-context answers.
    """
    pii_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]*?){13,16}\b')
    if pii_pattern.search(answer):
        logger.warning("Output Guardrail triggered: Potential PII detected.")
        return {"is_safe": False, "reason": "The answer contained sensitive personal information."}

    policy_leak_keywords = [
        "system prompt",
        "developer message",
        "hidden instructions",
        "internal policy",
        "ignore previous instructions",
    ]
    lowered = answer.lower()
    if any(keyword in lowered for keyword in policy_leak_keywords):
        logger.warning("Output Guardrail triggered: Potential instruction leakage.")
        return {"is_safe": False, "reason": "The answer attempted to reveal internal instructions."}

    if retrieved_chunks is not None and not retrieved_chunks:
        allowed = (
            "don't have enough information" in lowered
            or "do not have enough information" in lowered
            or "outside my curated" in lowered
        )
        if not allowed:
            logger.warning("Output Guardrail triggered: Answer produced without context.")
            return {"is_safe": False, "reason": "The answer was not grounded in retrieved sources."}

    return {"is_safe": True, "reason": ""}
