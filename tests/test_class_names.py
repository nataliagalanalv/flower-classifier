"""
Unit tests for the class index -> flower name mapping logic.
"""

from class_names import idx_to_name


def test_known_index_returns_a_name():
    """Index 0 (torchvision) maps to JSON key "1" — should return a real name, not the fallback."""
    name = idx_to_name(0)

    assert not name.startswith("clase_")


def test_offset_is_applied_correctly():
    """torchvision index 49 (petunia, per our analysis) should map to JSON key "50"."""
    name = idx_to_name(50)

    assert name == "petunia"


def test_unknown_index_falls_back_to_placeholder():
    """An out-of-range index should not raise, and should use the fallback format."""
    name = idx_to_name(9999)

    assert name == "clase_9999"