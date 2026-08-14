"""
Path resolution and dataset discovery utility for local and Kaggle environments.
"""

import os
from pathlib import Path
from typing import Optional


def get_base_dir() -> Path:
    """Returns the base project directory."""
    # If running inside a script or module
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_data_dir() -> Path:
    """
    Finds the competition data directory dynamically.
    Checks:
    1. Environment variable RSNA_KNEE_DATA_DIR
    2. Kaggle default input path: /kaggle/input/rsna-knee-abnormality-detection
    3. Local data directory: project_root/data
    """
    if "RSNA_KNEE_DATA_DIR" in os.environ:
        return Path(os.environ["RSNA_KNEE_DATA_DIR"])

    kaggle_path = Path("/kaggle/input/rsna-knee-abnormality-detection")
    if kaggle_path.exists():
        return kaggle_path

    # Fallback to local data folder
    local_data = get_base_dir() / "data"
    return local_data


def get_metadata_path(filename: str = "train.csv") -> Path:
    """Returns path to metadata CSV file."""
    data_dir = get_data_dir()
    direct_path = data_dir / filename
    if direct_path.exists():
        return direct_path
    
    metadata_subpath = data_dir / "metadata" / filename
    if metadata_subpath.exists():
        return metadata_subpath
    
    return direct_path


def get_output_dir() -> Path:
    """Returns outputs directory, ensuring it exists."""
    if Path("/kaggle/working").exists():
        out = Path("/kaggle/working/outputs")
    else:
        out = get_base_dir() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out
