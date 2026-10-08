"""
   Utility for mapping Flowers102 numeric class indices to human-readable names.

   The names file is bundled inside the package and loaded lazily, so importing
   this module has no side effects and works from any working directory.
"""

import json
from functools import lru_cache
from importlib.resources import files

@lru_cache(maxsize=1)
def load_cat_to_name():
    """Read the class-name mapping bundled inside the package (cached)."""
    resource = files("flower_classifier") / "resources" / "cat_to_name.json"
    return json.loads(resource.read_text(encoding="utf-8"))


def idx_to_name(idx):
       """Map a torchvision Flowers102 class index (0-101) to a flower name.
       The community JSON is 1-indexed ("1".."102"), while torchvision labels are 0-indexed, 
       hence the +1 offset. Falls back to a placeholder string instead of raising KeyError if the index is missing from the JSON.
       """
       return load_cat_to_name().get(str(idx + 1), f"clase_{idx}")