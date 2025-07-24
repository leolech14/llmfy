import pytest

from src.quality.quality_scorer_v2 import ImprovedQualityAnalyzer

@pytest.fixture
def analyzer():
    return ImprovedQualityAnalyzer()

@pytest.fixture
def sample_text():
    return (
        "Instead, top designers use off-black and tinted dark palettes. "
        "For example, Google's Material Design dark theme recommends a very dark gray (#121212) as the base surface color. "
        "This softer black reduces eye strain in low-light conditions and prevents the high contrast issues of pure black."
    )


def test_analyze_returns_expected_score(analyzer, sample_text):
    result = analyzer.analyze(sample_text)
    assert pytest.approx(result["overall_score"], 0.01) == 7.87
    assert set(result["strengths"]) == {"context_independence", "semantic_coherence", "clarity"}


def test_analyze_artifact_returns_zero(analyzer):
    result = analyzer.analyze("12345")
    assert result["overall_score"] == 0.0
