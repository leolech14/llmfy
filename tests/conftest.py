"""
Test configuration for llmfy package
"""
import pytest
from pathlib import Path

# Get project root for test fixtures and data
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def test_data_dir():
    """Directory containing test data files"""
    return ROOT / "tests" / "data"

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs"""
    return tmp_path
