import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROFILES_DIR = Path.home() / "Downloads" / "output_profiles"
DEFAULT_ESCO_SKILLS_PATH = Path.home() / "Downloads" / "mapping_of_ESCO_skills.xlsx"
DEFAULT_ESCO_OCCUPATIONS_PATH = Path.home() / "Downloads" / "mapping_of_ESCO_occupations.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRACKER_API_URL = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
SKILLAB_BASE_URL = TRACKER_API_URL.removesuffix("/api")

DEFAULT_MIN_SECTOR_YEAR_PROFILES = 50

GENERIC_SKILL_LABELS = {
    "communicate with customers",
    "communicate with others",
    "communicate with colleagues",
    "customer service",
    "manage tasks",
    "organise work",
    "solve problems",
    "teamwork principles",
    "use communication techniques",
    "work in teams",
}

PREDICTION_SCORE_WEIGHTS = {
    "profile_growth_score": 0.20,
    "recent_adoption_score": 0.10,
    "global_profile_growth_score": 0.15,
    "evidence_support_score": 0.15,
    "job_demand_score": 0.25,
    "job_demand_growth_score": 0.10,
    "sector_similarity_score": 0.05,
}
