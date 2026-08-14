"""
Unit tests for multilingual clinical report abnormality extraction, negation, and uncertainty parsing.
"""

import pytest
from rsna_knee.reports.extractor import ReportAbnormalityExtractor


@pytest.fixture
def extractor():
    return ReportAbnormalityExtractor()


def test_positive_acl_extraction(extractor):
    report = "IMPRESSION: Complete tear of the anterior cruciate ligament with adjacent bone contusion."
    res = extractor.extract_study_report(report)

    assert res["ACL"]["state"] == "positive"
    assert res["ACL"]["probability"] >= 0.90
    assert res["Contusion"]["state"] == "positive"
    assert res["Contusion"]["probability"] >= 0.90
    assert res["Fracture"]["state"] == "not_mentioned"


def test_negated_meniscus_extraction(extractor):
    report = "FINDINGS: Medial and lateral menisci are intact. No joint effusion or Baker's cyst. IMPRESSION: Normal knee MRI."
    res = extractor.extract_study_report(report)

    assert res["Medial Meniscus"]["state"] == "negative"
    assert res["Medial Meniscus"]["probability"] <= 0.10
    assert res["Lateral Meniscus"]["state"] == "negative"
    assert res["Lateral Meniscus"]["probability"] <= 0.10
    assert res["Effusion"]["state"] == "negative"
    assert res["Baker's"]["state"] == "negative"


def test_uncertain_finding_extraction(extractor):
    report = "FINDINGS: Possible subtle undisplaced tibial plateau fracture, cannot exclude occult injury."
    res = extractor.extract_study_report(report)

    assert res["Fracture"]["state"] == "uncertain"
    assert res["Fracture"]["probability"] == 0.50
    assert res["Fracture"]["loss_mask"] == False


def test_multilingual_german_extraction(extractor):
    report = "BEFUND: Deutliche VKB-Ruptur mit ausgeprägter Innenbandläsion und begleitendem Gelenkerguss. Keine Fraktur."
    res = extractor.extract_study_report(report)

    assert res["ACL"]["state"] == "positive"
    assert res["MCL"]["state"] == "positive"
    assert res["Effusion"]["state"] == "positive"
    assert res["Fracture"]["state"] == "negative"


def test_multilingual_spanish_extraction(extractor):
    report = "CONCLUSIÓN: Rotura del menisco interno. Sin signos de artrosis femoropatelar ni derrame articular."
    res = extractor.extract_study_report(report)

    assert res["Medial Meniscus"]["state"] == "positive"
    assert res["PF OA"]["state"] == "negative"
    assert res["Effusion"]["state"] == "negative"
