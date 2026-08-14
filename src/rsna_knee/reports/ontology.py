"""
Ontology loader and clinical keyword structures for knee abnormality labeling.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from rsna_knee.paths import get_base_dir


def load_ontology(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads target ontology definitions from configs/target_ontology.yaml.
    """
    if config_path is None:
        config_path = get_base_dir() / "configs" / "target_ontology.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Target ontology file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("targets", {})
