from pathlib import Path

import pandas as pd


PROFILE_COLUMNS = [
    "id",
    "country",
    "user_country",
    "region",
    "city",
    "location",
    "company",
    "skills",
    "occupations",
    "description",
    "content",
    "startdate",
    "source",
    "source_id",
]


def list_profile_files(profiles_dir, limit_files=None):
    files = sorted(Path(profiles_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {profiles_dir}")
    return files[:limit_files] if limit_files else files


def load_profiles(profiles_dir, limit_files=None):
    frames = []
    for file_path in list_profile_files(profiles_dir, limit_files=limit_files):
        df = pd.read_parquet(file_path)
        available_columns = [column for column in PROFILE_COLUMNS if column in df.columns]
        df = df[available_columns].copy()
        df["source_file"] = file_path.name
        df["profile_uid"] = file_path.stem + ":" + df["id"].astype(str)
        frames.append(df)

    profiles = pd.concat(frames, ignore_index=True)
    profiles["year"] = pd.to_datetime(profiles["startdate"], errors="coerce").dt.year
    profiles = profiles.dropna(subset=["year"])
    profiles["year"] = profiles["year"].astype(int)
    return profiles


def explode_profile_skills_occupations(profiles):
    base_columns = [
        "profile_uid",
        "year",
        "country",
        "user_country",
        "region",
        "city",
        "location",
        "company",
        "source",
        "source_file",
    ]
    available_base = [column for column in base_columns if column in profiles.columns]

    exploded = profiles[available_base + ["skills", "occupations"]].copy()
    exploded = exploded.explode("skills").rename(columns={"skills": "skill_uri"})
    exploded = exploded.explode("occupations").rename(columns={"occupations": "occupation_uri"})
    exploded = exploded.dropna(subset=["skill_uri", "occupation_uri"])
    exploded["skill_uri"] = exploded["skill_uri"].astype(str)
    exploded["occupation_uri"] = exploded["occupation_uri"].astype(str)

    return exploded.drop_duplicates(["profile_uid", "year", "skill_uri", "occupation_uri"])

