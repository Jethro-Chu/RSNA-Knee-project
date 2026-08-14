"""
Weak supervision label generator.
Processes train.csv radiology reports into structured, versioned pseudo-labels.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna_knee.constants import ID_COLUMN, TARGET_NAMES
from rsna_knee.reports.extractor import ReportAbnormalityExtractor


def generate_pseudo_labels_dataframe(
    train_df: pd.DataFrame,
    extractor: Optional[ReportAbnormalityExtractor] = None,
) -> pd.DataFrame:
    """
    Generates a structured pseudo-labels DataFrame with soft probabilities, states, and loss masks.
    """
    if extractor is None:
        extractor = ReportAbnormalityExtractor()

    records = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting Report Labels"):
        study_id = row[ID_COLUMN]
        report_text = str(row.get("Report", ""))

        extracted = extractor.extract_study_report(report_text)

        study_record = {ID_COLUMN: study_id}
        
        # Check if ground-truth expert labels exist for this row
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
            if has_expert and target in expert_labels:
                # Expert label takes 100% priority
                study_record[f"{target}_prob"] = expert_labels[target]
                study_record[f"{target}_state"] = "expert_positive" if expert_labels[target] == 1.0 else "expert_negative"
                study_record[f"{target}_confidence"] = 1.0
                study_record[f"{target}_loss_mask"] = True
                study_record[f"{target}_source"] = "expert"
            else:
                info = extracted[target]
                study_record[f"{target}_prob"] = info["probability"]
                study_record[f"{target}_state"] = info["state"]
                study_record[f"{target}_confidence"] = info["confidence"]
                study_record[f"{target}_loss_mask"] = info["loss_mask"]
                study_record[f"{target}_source"] = "nlp_report"

        records.append(study_record)

    pseudo_df = pd.DataFrame(records)
    return pseudo_df
