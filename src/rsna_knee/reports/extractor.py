"""
Clinical report abnormality extractor for RSNA Knee MRI reports.
Distinguishes 4 semantic states per abnormality:
  - 'positive' (p ~ 0.95, loss_mask=True, confidence=0.90)
  - 'negative' (p ~ 0.05, loss_mask=True, confidence=0.90)
  - 'uncertain' (p ~ 0.50, loss_mask=False)
  - 'not_mentioned' (p ~ 0.10, loss_mask=False)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from rsna_knee.constants import TARGET_NAMES
from rsna_knee.reports.normalization import normalize_report_text, segment_report_sections
from rsna_knee.reports.ontology import load_ontology


class ReportAbnormalityExtractor:
    """
    Deterministic multilingual NLP extractor for the 12 knee abnormalities.
    Handles section weighting, negation scoping, uncertainty qualifiers, proximity rules, and exclusions.
    """

    def __init__(self, ontology: Optional[Dict[str, Any]] = None):
        if ontology is None:
            ontology = load_ontology()
        self.ontology = ontology

        # Common negation patterns across languages (en, es, fr, de, hr, tr, el, bg)
        self.negation_patterns = [
            r"\bno\b",
            r"\bwithout\b",
            r"\bfree of\b",
            r"\bnegative for\b",
            r"\bno evidence of\b",
            r"\brules? out\b",
            r"\babsence of\b",
            r"\bintact\b",
            r"\bunremarkable\b",
            r"\bdenies\b",
            r"\bsin\b",
            r"\bsin signos de\b",
            r"\bni\b",
            r"\bsans\b",
            r"\bpas de\b",
            r"\bkein\b",
            r"\bkeine\b",
            r"\bkeinerlei\b",
            r"\bohne\b",
            r"\bausschluss\b",
            r"\bbez\b",
            r"\bnema\b",
            r"\byok\b",
            r"\bsaptanmadı\b",
            r"\bχωρίς\b",
            r"\bδεν\b",
            r"\bбез\b",
            r"\bняма\b",
        ]
        self.neg_regex = re.compile("|".join(self.negation_patterns), re.IGNORECASE)

        # Uncertainty qualifiers
        self.uncertainty_patterns = [
            r"\bpossible\b",
            r"\bsuspected\b",
            r"\bindeterminate\b",
            r"\bequivocal\b",
            r"\bquestionable\b",
            r"\bcannot (?:rule out|exclude)\b",
            r"\bposible\b",
            r"\bsospecha\b",
            r"\bsuspecté\b",
            r"\bmöglich\b",
            r"\bverdacht auf\b",
            r"\bvjerojatno\b",
            r"\bolası\b",
            r"\bπιθανόν\b",
        ]
        self.unc_regex = re.compile("|".join(self.uncertainty_patterns), re.IGNORECASE)

        # Proximity patterns for complex multi-word clinical concepts (e.g. OA, Menisci, Ligaments)
        self.proximity_rules = {
            "Medial OA": [
                re.compile(r"\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|gonartros\w*)\b.{1,40}\b(?:medial\w*|interno|medijaln\w*|femorotibial\s+medial)\b", re.IGNORECASE),
                re.compile(r"\b(?:medial\w*|interno|medijaln\w*)\b.{1,40}\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|gonartros\w*)\b", re.IGNORECASE),
            ],
            "Lateral OA": [
                re.compile(r"\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|gonartros\w*)\b.{1,40}\b(?:lateral\w*|externo|lateraln\w*|femorotibial\s+lateral)\b", re.IGNORECASE),
                re.compile(r"\b(?:lateral\w*|externo|lateraln\w*)\b.{1,40}\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|gonartros\w*)\b", re.IGNORECASE),
            ],
            "PF OA": [
                re.compile(r"\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|condropat\w*)\b.{1,40}\b(?:pf|patellofemoral\w*|patellar\w*|femoropatellar\w*|rotulian\w*|patele)\b", re.IGNORECASE),
                re.compile(r"\b(?:pf|patellofemoral\w*|patellar\w*|femoropatellar\w*|rotulian\w*|patele)\b.{1,40}\b(?:oa|osteoarthrit\w*|arthros\w*|artros\w*|chondros\w*|hondromalac\w*|chondromalac\w*|condropat\w*)\b", re.IGNORECASE),
            ],
        }

    def extract_study_report(self, report_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extracts states and probabilities for all 12 targets from a single report.
        """
        norm_text = normalize_report_text(report_text)
        sections = segment_report_sections(norm_text)

        # Focus diagnostic extraction primarily on findings and impression
        diagnostic_text = f"{sections['findings']} {sections['impression']}".strip()
        if not diagnostic_text:
            diagnostic_text = norm_text

        results: Dict[str, Dict[str, Any]] = {}

        for target in TARGET_NAMES:
            target_def = self.ontology.get(target, {})
            state, confidence, evidence = self._evaluate_target(target, target_def, diagnostic_text, norm_text)

            # Map semantic state to soft probability
            if state == "positive":
                prob = 0.95
                loss_mask = True
            elif state == "negative":
                prob = 0.05
                loss_mask = True
            elif state == "uncertain":
                prob = 0.50
                loss_mask = False  # Zero loss weight for training
            else:  # not_mentioned
                prob = 0.10
                loss_mask = False

            results[target] = {
                "state": state,
                "probability": prob,
                "confidence": confidence,
                "evidence": evidence,
                "loss_mask": loss_mask,
            }

        return results

    def _evaluate_target(
        self,
        target: str,
        target_def: Dict[str, Any],
        diagnostic_text: str,
        full_text: str,
    ) -> Tuple[str, float, str]:
        """
        Evaluates presence of a single target within text.
        """
        def _flatten_terms(terms_obj) -> List[str]:
            if isinstance(terms_obj, dict):
                res = []
                for val in terms_obj.values():
                    if isinstance(val, list):
                        res.extend([str(t).lower() for t in val])
                return res
            elif isinstance(terms_obj, list):
                return [str(t).lower() for t in terms_obj]
            return []

        pos_terms = _flatten_terms(target_def.get("positive_terms", {}))
        neg_terms = _flatten_terms(target_def.get("negative_terms", {}))
        unc_terms = _flatten_terms(target_def.get("uncertain_terms", {}))
        exclusions = _flatten_terms(target_def.get("exclusions", {}))

        # 1. Check explicit negative terms first
        for neg_term in neg_terms:
            if neg_term in diagnostic_text:
                return "negative", 0.95, f"Explicit negative term: '{neg_term}'"

        # 2. Check positive terms
        matched_positive = None
        for pos_term in pos_terms:
            if pos_term in diagnostic_text:
                # Check if preceded by negation within local context window (up to 80 chars before)
                idx = diagnostic_text.find(pos_term)
                context_window = diagnostic_text[max(0, idx - 80):idx]
                if self.neg_regex.search(context_window):
                    return "negative", 0.90, f"Negated positive term: '{context_window} {pos_term}'"
                
                # Check if preceded by uncertainty
                if self.unc_regex.search(context_window):
                    return "uncertain", 0.60, f"Uncertain positive term: '{context_window} {pos_term}'"

                matched_positive = pos_term
                break

        # 3. Check proximity rules if no direct term match
        if not matched_positive and target in self.proximity_rules:
            for pattern in self.proximity_rules[target]:
                m = pattern.search(diagnostic_text)
                if m:
                    span_start = m.start()
                    context_window = diagnostic_text[max(0, span_start - 80):span_start]
                    if self.neg_regex.search(context_window):
                        return "negative", 0.90, f"Negated proximity match: '{m.group()}'"
                    if self.unc_regex.search(context_window):
                        return "uncertain", 0.60, f"Uncertain proximity match: '{m.group()}'"
                    matched_positive = f"proximity: {m.group()}"
                    break

        if matched_positive:
            # Check exclusions
            for exc in exclusions:
                if exc in diagnostic_text:
                    return "negative", 0.85, f"Exclusion matched: '{exc}'"
            return "positive", 0.95, f"Positive match: '{matched_positive}'"

        # 4. Check uncertain terms
        for unc_term in unc_terms:
            if unc_term in diagnostic_text:
                return "uncertain", 0.60, f"Uncertain match: '{unc_term}'"

        return "not_mentioned", 0.50, "Not mentioned in report"
