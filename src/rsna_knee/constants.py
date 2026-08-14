"""
Constants and target definitions for RSNA Knee Abnormality Detection.
"""

from typing import List

# Official 12 target column names and exact canonical ordering
TARGET_NAMES: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Alias dictionary to handle possible naming variations in competition artifacts
TARGET_ALIASES = {
    "Synovitis": "Synovitis",
    "Synvitis": "Synovitis",
    "Synvitis": "Synvitis",
    "Baker's": "Baker's",
    "Baker's Cyst": "Baker's",
    "Bakers Cyst": "Baker's",
    "Medial Meniscus Tear": "Medial Meniscus",
    "Lateral Meniscus Tear": "Lateral Meniscus",
    "Medial Osteoarthritis": "Medial OA",
    "Lateral Osteoarthritis": "Lateral OA",
    "Patellofemoral Osteoarthritis": "PF OA",
    "Bone Contusion": "Contusion",
    "Joint Effusion": "Effusion",
}

# Anatomical Planes
PLANES = ["sagittal", "coronal", "axial", "unknown"]

# Expected Submission Column Schema
ID_COLUMN = "StudyInstanceUID"
SUBMISSION_COLUMNS = [ID_COLUMN] + TARGET_NAMES
