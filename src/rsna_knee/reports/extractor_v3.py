"""
Clinical Report Abnormality Extractor v3 for RSNA Knee MRI Reports.
Features:
- Multi-tier semantic states: definite_positive, probable_positive, possible_positive, explicit_negative, not_mentioned, conflict.
- Strict clause-local negation with punctuation and contrast conjunction boundaries.
- Compartment and anatomical entity binding (Medial vs Lateral Meniscus, Medial vs Lateral Tibiofemoral OA, Patellofemoral OA).
- Expanded multilingual support (English, Spanish, French, German, Croatian, Greek, Dutch).
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rsna_knee.constants import TARGET_NAMES


def strip_accents(text: str) -> str:
    """Removes diacritics and accents for robust multilingual string matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", text).lower()


class ReportAbnormalityExtractorV3:
    """
    Deterministic NLP Extractor v3 with clause segmentation, multi-tier evidence weighting,
    and cross-compartment isolation.
    """

    def __init__(self, ontology_path: Optional[Path] = None):
        if ontology_path is None:
            ontology_path = Path(__file__).parent / "ontology_v3.json"
        
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        # Negation keywords across languages (contextualized to prevent false negation of 'without collapse')
        self.negation_terms = [
            r"\bno\b", r"\bwithout (?:evidence of|abnormality|abnormalities|tear|tears|rupture|fracture|effusion|defect|edema|fluid|lesion|stenosis|displacement|spur|spurring)\b",
            r"\bfree of\b", r"\bnegative for\b", r"\bno evidence of\b",
            r"\brules? out\b", r"\babsence of\b", r"\bunremarkable\b", r"\bdenies\b", r"\bintact\b",
            r"\bsin\b", r"\bsin signos de\b", r"\bni\b", r"\bsans\b", r"\bpas de\b",
            r"\bkein\b", r"\bkeine\b", r"\bkeinerlei\b", r"\bohne (?:befund|ruptur|erguss|fraktur|lasion|hinweis)\b", r"\bausschluss\b",
            r"\bbez\b", r"\bnema\b", r"\byok\b", r"\bsaptanmadi\b", r"\bχωρις\b", r"\bδεν\b",
            r"\bбез\b", r"\bняμα\b"
        ]
        self.neg_regex = re.compile("|".join(self.negation_terms), re.IGNORECASE)

        # Uncertainty qualifiers
        self.uncertainty_terms = [
            r"\bpossible\b", r"\bsuspected\b", r"\bindeterminate\b", r"\bequivocal\b",
            r"\bquestionable\b", r"\bcannot (?:rule out|exclude)\b", r"\bposible\b",
            r"\bsospecha\b", r"\bsuspecte\b", r"\bmoglich\b", r"\bverdacht auf\b",
            r"\bvjerojatno\b", r"\bmoguca\b", r"\bolasi\b", r"\bπιθανον\b"
        ]
        self.unc_regex = re.compile("|".join(self.uncertainty_terms), re.IGNORECASE)

        # Contrast boundary terms that reset negation
        self.contrast_boundary_regex = re.compile(
            r"\b(?:but|however|although|while|whereas|nevertheless|pero|sin embargo|mais|aber|jedoch|ali|ali i)\b",
            re.IGNORECASE
        )

        # Compile regexes per target
        self._compiled_rules = {}
        for target, data in self.ontology.items():
            self._compiled_rules[target] = {
                "anatomy": self._compile_term_list(self._get_all_anatomy(data)),
                "definite": self._compile_term_list(data.get("definite_positive_terms", [])),
                "probable": self._compile_term_list(data.get("probable_positive_terms", [])),
                "possible": self._compile_term_list(data.get("possible_positive_terms", [])),
                "negative": self._compile_term_list(data.get("negative_terms", [])),
                "compartment": data.get("compartment", None)
            }

    def _get_all_anatomy(self, data: Dict[str, Any]) -> List[str]:
        terms = []
        anat = data.get("anatomy_terms", {})
        for lang_terms in anat.values():
            terms.extend(lang_terms)
        return terms

    def _compile_term_list(self, term_list: List[str]) -> List[Tuple[re.Pattern, str]]:
        compiled = []
        # Sort by length descending to match longest phrases first
        for t in sorted(set(term_list), key=len, reverse=True):
            norm_t = strip_accents(t).replace("-", " ")
            # Create regex with word boundaries
            pattern = re.compile(r"(?<!\w)" + re.escape(norm_t) + r"(?!\w)", re.IGNORECASE)
            compiled.append((pattern, t))
        return compiled

    def segment_clauses(self, text: str) -> List[str]:
        """
        Segments text into isolated diagnostic clauses based on sentence boundaries,
        semicolons, newlines, and strong contrast conjunctions.
        """
        # Normalize hyphens and dashes to spaces for unified matching
        text = re.sub(r"[-–—/]+", " ", text)
        
        # Replace newlines, semicolons, and periods with explicit clause delimiters
        text = re.sub(r"[\n\r]+", " [CLAUSE_BREAK] ", text)
        text = re.sub(r"[;\u2022\u2023\u25E6\u2043\u2219]+", " [CLAUSE_BREAK] ", text)
        text = re.sub(r"\.\s+", " [CLAUSE_BREAK] ", text)
        
        # Split on contrast boundaries
        text = self.contrast_boundary_regex.sub(" [CLAUSE_BREAK] ", text)

        clauses = [c.strip() for c in text.split("[CLAUSE_BREAK]") if len(c.strip()) > 1]
        return clauses

    def extract_study_report(self, report_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extracts multi-tier states, probabilities, confidence, and loss weights for all 12 targets.
        """
        norm_full = strip_accents(report_text)
        clauses = self.segment_clauses(norm_full)

        results: Dict[str, Dict[str, Any]] = {}

        for target in TARGET_NAMES:
            rule = self._compiled_rules[target]
            target_eval = self._evaluate_target_across_clauses(target, rule, clauses, norm_full)
            results[target] = target_eval

        return results

    def _evaluate_target_across_clauses(
        self,
        target: str,
        rule: Dict[str, Any],
        clauses: List[str],
        full_text: str
    ) -> Dict[str, Any]:
        """
        Evaluates presence of evidence for a specific target across all segmented clauses.
        """
        evidence_list = []
        compartment = rule["compartment"]

        for clause in clauses:
            # 1. Check if clause mentions target anatomy or is compartment-bound
            has_anatomy, matched_anat = self._matches_any(rule["anatomy"], clause)
            
            # Check compartment interference (e.g. lateral mention in medial target rule)
            if compartment == "medial":
                if ("lateral" in clause or "externo" in clause or "aussen" in clause) and not ("medial" in clause or "interno" in clause or "innen" in clause):
                    continue # Clause belongs to lateral compartment
            elif compartment == "lateral":
                if ("medial" in clause or "interno" in clause or "innen" in clause) and not ("lateral" in clause or "externo" in clause or "aussen" in clause):
                    continue # Clause belongs to medial compartment
            elif compartment == "patellofemoral":
                if ("tibiofemoral" in clause) and not ("patell" in clause or "rotul" in clause or "trochl" in clause or "pf" in clause):
                    continue

            # 2. If target has specific anatomy, require anatomy mention unless target itself is anatomical (e.g. ACL/MCL/Baker's)
            if not has_anatomy:
                # For OA and Meniscus, check if compartment is clearly stated along with finding
                if compartment is not None:
                    has_comp_term = False
                    if compartment == "medial" and ("medial" in clause or "interno" in clause or "innen" in clause or "eso" in clause):
                        has_comp_term = True
                    elif compartment == "lateral" and ("lateral" in clause or "externo" in clause or "aussen" in clause or "ekso" in clause):
                        has_comp_term = True
                    elif compartment == "patellofemoral" and ("patell" in clause or "rotul" in clause or "trochl" in clause or "pf" in clause):
                        has_comp_term = True
                    
                    if not has_comp_term:
                        continue
                else:
                    continue

            # 3. Check for Negation in this specific clause
            is_negated = bool(self.neg_regex.search(clause))
            is_uncertain = bool(self.unc_regex.search(clause))
            has_explicit_neg, neg_term = self._matches_any(rule["negative"], clause)

            if is_negated or has_explicit_neg:
                # Verify negation isn't negated itself or false alarm
                evidence_list.append({
                    "tier": "explicit_negative",
                    "clause": clause,
                    "matched": neg_term if has_explicit_neg else "negated_clause"
                })
                continue

            # 4. Check Positive Tiers (Definite > Probable > Possible)
            has_def, def_term = self._matches_any(rule["definite"], clause)
            if has_def:
                tier = "possible_positive" if is_uncertain else "definite_positive"
                evidence_list.append({"tier": tier, "clause": clause, "matched": def_term})
                continue

            has_prob, prob_term = self._matches_any(rule["probable"], clause)
            if has_prob:
                tier = "possible_positive" if is_uncertain else "probable_positive"
                evidence_list.append({"tier": tier, "clause": clause, "matched": prob_term})
                continue

            has_poss, poss_term = self._matches_any(rule["possible"], clause)
            if has_poss:
                evidence_list.append({"tier": "possible_positive", "clause": clause, "matched": poss_term})
                continue

        # Resolve Final State from accumulated evidence
        return self._resolve_evidence(evidence_list, target)

    def _matches_any(self, pattern_list: List[Tuple[re.Pattern, str]], text: str) -> Tuple[bool, Optional[str]]:
        for pattern, raw_term in pattern_list:
            m = pattern.search(text)
            if m:
                return True, raw_term
        return False, None

    def _resolve_evidence(self, evidence_list: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        if not evidence_list:
            return {
                "state": "not_mentioned",
                "probability": 0.10,
                "confidence": 0.10,
                "loss_mask": False,
                "loss_weight": 0.0,
                "tier": "not_mentioned",
                "evidence": ""
            }

        tiers = [e["tier"] for e in evidence_list]
        has_positive = any("positive" in t for t in tiers)
        has_negative = any("negative" in t for t in tiers)

        if has_positive and has_negative:
            # Conflict: check if definite positive overrides or assign conflict
            if "definite_positive" in tiers:
                chosen_tier = "definite_positive"
            else:
                return {
                    "state": "conflict",
                    "probability": 0.50,
                    "confidence": 0.30,
                    "loss_mask": False,
                    "loss_weight": 0.0,
                    "tier": "conflict",
                    "evidence": "Conflicting positive and negative clauses"
                }
        elif has_positive:
            if "definite_positive" in tiers:
                chosen_tier = "definite_positive"
            elif "probable_positive" in tiers:
                chosen_tier = "probable_positive"
            else:
                chosen_tier = "possible_positive"
        else:
            chosen_tier = "explicit_negative"

        # Multi-Tier Semantic Policy
        if chosen_tier == "definite_positive":
            prob = 0.98
            conf = 0.95
            weight = 1.0
            mask = True
            state = "positive"
        elif chosen_tier == "probable_positive":
            prob = 0.90
            conf = 0.85
            weight = 0.85
            mask = True
            state = "positive"
        elif chosen_tier == "possible_positive":
            prob = 0.70
            conf = 0.65
            weight = 0.40
            mask = True
            state = "positive"
        else: # explicit_negative
            prob = 0.02
            conf = 0.95
            weight = 1.0
            mask = True
            state = "negative"

        matched_spans = " | ".join([f"{e['tier']}: {e['matched']}" for e in evidence_list])

        return {
            "state": state,
            "probability": prob,
            "confidence": conf,
            "loss_mask": mask,
            "loss_weight": weight,
            "tier": chosen_tier,
            "evidence": matched_spans
        }
