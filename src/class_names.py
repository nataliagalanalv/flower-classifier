"""
Utility for mapping Flowers102 numeric class indices to human-readable names.

Extracted into its own module (rather than living inline in
confusion_analysis.py) so it can be imported and unit tested without
triggering that script's model loading and full test-set inference.
"""

import json

with open("data/cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)


def idx_to_name(idx):
    """Map a torchvision Flowers102 class index (0-101) to a flower name.

    The community JSON is 1-indexed ("1".."102"), while torchvision labels
    are 0-indexed, hence the +1 offset. Falls back to a placeholder string
    instead of raising KeyError if the index is missing from the JSON.
    """
    return cat_to_name.get(str(idx + 1), f"clase_{idx}")