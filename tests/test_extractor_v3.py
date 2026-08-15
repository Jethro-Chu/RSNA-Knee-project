"""
Unit tests for Clinical Report Abnormality Extractor v3.
Verifies clause-local negation, contrast boundaries, compartment binding, and multilingual abbreviations.
"""

import pytest
from rsna_knee.reports.extractor_v3 import ReportAbnormalityExtractorV3


@pytest.fixture
def extractor():
    return ReportAbnormalityExtractorV3()


def test_required_semicolon_compartment_negation_isolation(extractor):
    """
    CRITICAL TEST CASE:
    'No focal chondrosis in the medial compartment; in the lateral compartment there is a cartilage defect.'
    Expected:
    - Medial OA: explicit negative or no positive evidence
    - Lateral OA: positive evidence
    - Negation must not cross the semicolon/anatomical boundary
    """
    text = "No focal chondrosis in the medial compartment; in the lateral compartment there is a cartilage defect."
    parsed = extractor.extract_study_report(text)

    medial_oa = parsed["Medial OA"]
    lateral_oa = parsed["Lateral OA"]

    assert medial_oa["state"] == "negative", f"Expected Medial OA to be negative, got {medial_oa}"
    assert lateral_oa["state"] == "positive", f"Expected Lateral OA to be positive, got {lateral_oa}"
    assert "cartilage defect" in lateral_oa["evidence"]


def test_spanish_acronym_expansion(extractor):
    """Verifies LCA -> ACL and LCM -> MCL in Spanish clinical reports."""
    text = "Resultados: Rotura completa del LCA. Rotura parcial del LCM. Meniscos intactos."
    parsed = extractor.extract_study_report(text)

    assert parsed["ACL"]["state"] == "positive"
    assert parsed["ACL"]["tier"] == "definite_positive"
    assert parsed["MCL"]["state"] == "positive"
    assert parsed["MCL"]["tier"] == "probable_positive"
    assert parsed["Medial Meniscus"]["state"] == "negative"
    assert parsed["Lateral Meniscus"]["state"] == "negative"


def test_contrast_conjunction_boundary(extractor):
    """Verifies that contrast terms like 'but' reset negation scope."""
    text = "The anterior cruciate ligament is intact, but there is a complete tear of the medial collateral ligament."
    parsed = extractor.extract_study_report(text)

    assert parsed["ACL"]["state"] == "negative"
    assert parsed["MCL"]["state"] == "positive"
    assert parsed["MCL"]["tier"] == "definite_positive"


def test_meniscal_compartment_isolation(extractor):
    """Verifies Medial vs Lateral meniscus tear assignment."""
    text = "Complex tear involving the posterior horn of the medial meniscus. Lateral meniscus appears intact without tear."
    parsed = extractor.extract_study_report(text)

    assert parsed["Medial Meniscus"]["state"] == "positive"
    assert parsed["Medial Meniscus"]["tier"] == "definite_positive"
    assert parsed["Lateral Meniscus"]["state"] == "negative"


def test_patellofemoral_oa_croatian(extractor):
    """Verifies PF OA detection in Croatian report text."""
    text = "PF artrotske promjene s reduciranim zglobnim prostorom i hondromalacijom IV stupnja."
    parsed = extractor.extract_study_report(text)

    assert parsed["PF OA"]["state"] == "positive"
    assert parsed["PF OA"]["tier"] == "definite_positive"


def test_effusion_and_bakers_cyst(extractor):
    """Verifies effusion and Baker's cyst detection."""
    text = "Large joint effusion with associated 3.5 cm Baker's cyst in the popliteal fossa. No fracture."
    parsed = extractor.extract_study_report(text)

    assert parsed["Effusion"]["state"] == "positive"
    assert parsed["Effusion"]["tier"] == "definite_positive"
    assert parsed["Baker's"]["state"] == "positive"
    assert parsed["Fracture"]["state"] == "negative"
