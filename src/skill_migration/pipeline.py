from pathlib import Path

from .config import (
    DEFAULT_ESCO_OCCUPATIONS_PATH,
    DEFAULT_ESCO_SKILLS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROFILES_DIR,
)
from .diffusion import (
    build_diffusion_events,
    detect_sector_entries,
    diffusion_leaderboard,
    predict_next_sectors,
)
from .demand import fetch_job_demand_signals
from .esco import load_occupation_mapping, load_skill_mapping
from .features import aggregate_skill_sector_year, build_profile_skill_sector_table
from .profiles import explode_profile_skills_occupations, load_profiles
from .reporting import generate_jury_artifacts


def run_pipeline(
    profiles_dir=DEFAULT_PROFILES_DIR,
    skills_path=DEFAULT_ESCO_SKILLS_PATH,
    occupations_path=DEFAULT_ESCO_OCCUPATIONS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    limit_files=5,
    min_mentions=2,
    min_adoption_rate=0.005,
    require_stability=False,
    use_skillab_demand=False,
    skillab_username=None,
    skillab_password=None,
    skillab_token=None,
    demand_limit=500,
    demand_source=None,
    demand_current_from_date=None,
    demand_current_to_date=None,
    demand_baseline_from_date=None,
    demand_baseline_to_date=None,
    demand_timeout=60,
    generate_artifacts=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading profile parquet files...")
    profiles = load_profiles(profiles_dir, limit_files=limit_files)
    print(f"Loaded {len(profiles):,} profiles")

    print("Loading ESCO mappings...")
    skill_mapping = load_skill_mapping(skills_path)
    occupation_mapping = load_occupation_mapping(occupations_path)
    print(f"Loaded {len(skill_mapping):,} skills and {len(occupation_mapping):,} occupations")

    print("Exploding skills and occupations...")
    profiles_long = explode_profile_skills_occupations(profiles)
    print(f"Built {len(profiles_long):,} profile-skill-occupation rows")

    print("Joining labels and sector proxies...")
    profile_skill_sector = build_profile_skill_sector_table(
        profiles_long,
        skill_mapping,
        occupation_mapping,
    )

    print("Aggregating skill adoption by year and sector...")
    skill_sector_year = aggregate_skill_sector_year(profile_skill_sector)

    print("Detecting sector entries and diffusion events...")
    entries = detect_sector_entries(
        skill_sector_year,
        min_mentions=min_mentions,
        min_adoption_rate=min_adoption_rate,
        require_stability=require_stability,
    )
    diffusion_events = build_diffusion_events(entries)
    leaderboard = diffusion_leaderboard(diffusion_events, skill_sector_year)

    job_demand = None
    if use_skillab_demand:
        print("Fetching Skillab job demand signals...")
        job_demand = fetch_job_demand_signals(
            username=skillab_username,
            password=skillab_password,
            token=skillab_token,
            limit=demand_limit,
            source=demand_source,
            current_from_date=demand_current_from_date,
            current_to_date=demand_current_to_date,
            baseline_from_date=demand_baseline_from_date,
            baseline_to_date=demand_baseline_to_date,
            timeout=demand_timeout,
        )
        job_demand.to_csv(output_dir / "job_demand_signals.csv", index=False)
        print(f"Loaded {len(job_demand):,} job demand skill signals")

    predictions = predict_next_sectors(
        skill_sector_year,
        entries,
        job_demand=job_demand,
    )

    profile_skill_sector.to_parquet(output_dir / "profile_skill_sector.parquet", index=False)
    skill_sector_year.to_parquet(output_dir / "skill_sector_year.parquet", index=False)
    entries.to_csv(output_dir / "sector_entries.csv", index=False)
    diffusion_events.to_csv(output_dir / "diffusion_events.csv", index=False)
    leaderboard.to_csv(output_dir / "diffusion_leaderboard.csv", index=False)
    predictions.to_csv(output_dir / "next_sector_predictions.csv", index=False)

    artifacts_dir = None
    if generate_artifacts:
        print("Generating jury artefacts...")
        artifacts_dir = generate_jury_artifacts(
            output_dir,
            profiles,
            profile_skill_sector,
            skill_sector_year,
            entries,
            diffusion_events,
            leaderboard,
            predictions,
            job_demand=job_demand,
            params={
                "min_mentions": min_mentions,
                "min_adoption_rate": min_adoption_rate,
                "require_stability": require_stability,
            },
        )

    print(f"Outputs written to {output_dir}")
    if artifacts_dir:
        print(f"Jury artefacts written to {artifacts_dir}")
    print_summary(leaderboard, diffusion_events, predictions)

    return {
        "profiles": profiles,
        "profile_skill_sector": profile_skill_sector,
        "skill_sector_year": skill_sector_year,
        "entries": entries,
        "diffusion_events": diffusion_events,
        "leaderboard": leaderboard,
        "predictions": predictions,
        "job_demand": job_demand,
        "artifacts_dir": artifacts_dir,
    }


def print_summary(leaderboard, diffusion_events, predictions):
    print("\nTop diffusing skills:")
    if leaderboard.empty:
        print("No diffusion events found with the current thresholds.")
    else:
        print(
            leaderboard[
                [
                    "skill_label",
                    "origin_sector",
                    "origin_year",
                    "sectors_reached",
                    "diffusion_score",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("\nExample diffusion events:")
    if diffusion_events.empty:
        print("No migration events found.")
    else:
        print(
            diffusion_events[
                [
                    "skill_label",
                    "origin_sector",
                    "origin_year",
                    "entered_sector",
                    "entered_year",
                    "delay_years",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("\nExample predicted next sectors:")
    if predictions.empty:
        print("No predictions available.")
    else:
        print(
            predictions.sort_values("prediction_score", ascending=False)[
                [
                    "skill_label",
                    "candidate_sector",
                    "prediction_score",
                    "evidence_support_score",
                    "job_demand_score",
                    "sector_similarity_score",
                    "recent_adoption_rate",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
