"""
Weak Supervision Label Generator v3 with Multi-Tier Evidence Schema for RSNA Knee MRI.
Generates versioned pseudo-labels with fine-grained confidence tiers and loss weighting.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES
from rsna_knee.reports.extractor_v3 import ReportAbnormalityExtractorV3


def generate_pseudo_labels_v3_dataframe(
    train_df: pd.DataFrame,
    extractor: Optional[ReportAbnormalityExtractorV3] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates structured multi-tier pseudo-labels DataFrame v3 and full evidence audit DataFrame.
    """
    if extractor is None:
        extractor = ReportAbnormalityExtractorV3()

    records = []
    evidence_records = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting Pseudo-Labels v3"):
        study_id = row[ID_COLUMN]
        report_text = str(row.get("Report", ""))

        extracted = extractor.extract_study_report(report_text)

        study_record = {ID_COLUMN: study_id}
        
        # Check for expert gold annotations
        has_expert = False
        expert_labels = {}
        for target in TARGET_NAMES:
            if target in row and not pd.isna(row[target]) and str(row[target]).strip() != "":
                try:
                    val = float(row[target])
                    expert_labels[target] = val
                    has_expert = True
                except Exception:
                    pass

        study_record["has_expert_labels"] = has_expert

        for target in TARGET_NAMES:
            info = extracted[target]
            
            # Evidence audit row
            evidence_records.append({
                "StudyInstanceUID": study_id,
                "target": target,
                "tier": info["tier"],
                "state": info["state"],
                "probability": info["probability"],
                "confidence": info["confidence"],
                "loss_weight": info["loss_weight"],
                "loss_mask": info["loss_mask"],
                "evidence": info["evidence"],
                "has_expert": has_expert,
                "expert_label": expert_labels.get(target, np.nan) if has_expert else np.nan,
                "parser_version": "v3.0"
            })

            if has_expert and target in expert_labels:
                # Expert gold label retains top priority and weight (w = 5.0)
                study_record[f"{target}_prob"] = expert_labels[target]
                study_record[f"{target}_state"] = "expert_positive" if expert_labels[target] == 1.0 else "expert_negative"
                study_record[f"{target}_tier"] = "expert_gold"
                study_record[f"{target}_confidence"] = 1.0
                study_record[f"{target}_loss_mask"] = True
                study_record[f"{target}_loss_weight"] = 5.0
                study_record[f"{target}_source"] = "expert_gold"
                study_record[f"{target}_evidence"] = "Expert human radiologist annotation"
            else:
                study_record[f"{target}_prob"] = info["probability"]
                study_record[f"{target}_state"] = info["state"]
                study_record[f"{target}_tier"] = info["tier"]
                study_record[f"{target}_confidence"] = info["confidence"]
                study_record[f"{target}_loss_mask"] = info["loss_mask"]
                study_record[f"{target}_loss_weight"] = info["loss_weight"]
                study_record[f"{target}_source"] = "nlp_v3"
                study_record[f"{target}_evidence"] = info["evidence"]

        records.append(study_record)

    pseudo_df = pd.DataFrame(records)
    evidence_df = pd.DataFrame(evidence_records)
    return pseudo_df, evidence_df
