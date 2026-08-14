"""
Text normalization, section parsing, and negation handling for radiology reports.
"""

import re
import unicodedata
from typing import Dict, List, Tuple


def normalize_report_text(text: str) -> str:
    """
    Normalizes unicode characters, replaces non-breaking spaces, and standardizes punctuation.
    """
    if not isinstance(text, str):
        return ""
    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Replace multiple whitespaces and newlines
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def segment_report_sections(text: str) -> Dict[str, str]:
    """
    Extracts structured sections from a radiology report (Clinical Indication, Technique, Findings, Impression).
    Prioritizes Findings and Impression for diagnostic extraction.
    """
    text_lower = text.lower()
    sections: Dict[str, str] = {
        "indication": "",
        "technique": "",
        "findings": "",
        "impression": "",
        "full_text": text,
    }

    # Common section header patterns in multilingual reports (en, es, fr, de)
    patterns = {
        "indication": r"(?:clinical history|indication|history|motivo de consulta|indication clinique|klinische angaben)[\s:]+",
        "technique": r"(?:technique|protocol|técnica|protocole|untersuchungstechnik)[\s:]+",
        "findings": r"(?:findings|befund|hallazgos|résultats)[\s:]+",
        "impression": r"(?:impression|conclusion|beurteilung|conclusión|conclusion)[\s:]+",
    }

    # Find section start indices
    matches = []
    for sec_name, pat in patterns.items():
        for m in re.finditer(pat, text_lower):
            matches.append((m.start(), m.end(), sec_name))

    matches.sort(key=lambda x: x[0])

    if not matches:
        sections["findings"] = text
        sections["impression"] = text
        return sections

    for i, (start, end, sec_name) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[end:next_start].strip()
        sections[sec_name] = content

    return sections
