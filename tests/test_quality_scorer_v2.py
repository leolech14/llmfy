from src.quality.quality_scorer_v2 import ImprovedQualityAnalyzer


def test_basic_quality_score():
    analyzer = ImprovedQualityAnalyzer()
    text = "This chunk contains useful information about the pipeline."
    result = analyzer.analyze(text)
    assert result["overall_score"] > 0
