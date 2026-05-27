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
    parser.add_argument("--min-mentions", type=int, default=2)
    parser.add_argument("--min-adoption-rate", type=float, default=0.005)
    parser.add_argument(
        "--min-sector-year-profiles",
        type=int,
        default=50,
        help="Minimum profiles required in a sector-year before an entry can be detected.",
    )
    parser.add_argument(
        "--require-stability",
        action="store_true",
        help="Require an entry threshold to be met in two consecutive years.",
    )
    parser.add_argument(
        "--exclude-generic-skills",
        action="store_true",
        help="Exclude broad generic skills from leaderboard and next-sector predictions.",
    )
    parser.add_argument(
        "--use-skillab-demand",
        action="store_true",
        help="Blend Skillab job demand analytics into next-sector predictions.",
    )
    parser.add_argument("--skillab-username")
    parser.add_argument("--skillab-password")
    parser.add_argument("--skillab-token")
    parser.add_argument("--demand-limit", type=int, default=500)
    parser.add_argument("--demand-source")
    parser.add_argument("--demand-current-from-date")
    parser.add_argument("--demand-current-to-date")
    parser.add_argument("--demand-baseline-from-date")
    parser.add_argument("--demand-baseline-to-date")
    parser.add_argument("--demand-timeout", type=int, default=60)
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip report, pitch outline, sensitivity tables, and SVG charts.",
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
        min_sector_year_profiles=args.min_sector_year_profiles,
        require_stability=args.require_stability,
        exclude_generic_skills=args.exclude_generic_skills,
        use_skillab_demand=args.use_skillab_demand,
        skillab_username=args.skillab_username,
        skillab_password=args.skillab_password,
        skillab_token=args.skillab_token,
        demand_limit=args.demand_limit,
        demand_source=args.demand_source,
        demand_current_from_date=args.demand_current_from_date,
        demand_current_to_date=args.demand_current_to_date,
        demand_baseline_from_date=args.demand_baseline_from_date,
        demand_baseline_to_date=args.demand_baseline_to_date,
        demand_timeout=args.demand_timeout,
        generate_artifacts=not args.skip_artifacts,
    )


if __name__ == "__main__":
    main()
