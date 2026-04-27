import re
import logging

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