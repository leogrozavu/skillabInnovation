import argparse

from src.skill_migration.config import (
    DEFAULT_ESCO_OCCUPATIONS_PATH,
    DEFAULT_ESCO_SKILLS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROFILES_DIR,
)
from src.skill_migration.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a skill migration map from ESCO profile data."
    )
    parser.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--skills-path", default=DEFAULT_ESCO_SKILLS_PATH)
    parser.add_argument("--occupations-path", default=DEFAULT_ESCO_OCCUPATIONS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit-files",
        type=int,
        default=5,
        help="Number of profile parquet files to process. Use 0 for all files.",
    )
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--min-adoption-rate", type=float, default=0.005)
    parser.add_argument(
        "--require-stability",
        action="store_true",
        help="Require an entry threshold to be met in two consecutive years.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        profiles_dir=args.profiles_dir,
        skills_path=args.skills_path,
        occupations_path=args.occupations_path,
        output_dir=args.output_dir,
        limit_files=args.limit_files or None,
        min_mentions=args.min_mentions,
        min_adoption_rate=args.min_adoption_rate,
        require_stability=args.require_stability,
    )


if __name__ == "__main__":
    main()
