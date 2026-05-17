import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROFILES_DIR = Path.home() / "Downloads" / "output_profiles"
DEFAULT_ESCO_SKILLS_PATH = Path.home() / "Downloads" / "mapping_of_ESCO_skills.xlsx"
DEFAULT_ESCO_OCCUPATIONS_PATH = Path.home() / "Downloads" / "mapping_of_ESCO_occupations.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRACKER_API_URL = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
SKILLAB_BASE_URL = TRACKER_API_URL.removesuffix("/api")
