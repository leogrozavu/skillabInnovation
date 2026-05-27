import pandas as pd

from .config import GENERIC_SKILL_LABELS


def build_profile_skill_sector_table(profiles_long, skill_mapping, occupation_mapping):
    df = profiles_long.merge(skill_mapping, on="skill_uri", how="left")
    df = df.merge(
        occupation_mapping[
            ["occupation_uri", "occupation_label", "isco_code", "sector_proxy"]
        ],
        on="occupation_uri",
        how="left",
    )
    df["skill_label"] = df["skill_label"].fillna(df["skill_uri"])
    df["occupation_label"] = df["occupation_label"].fillna(df["occupation_uri"])
    df["sector_proxy"] = df["sector_proxy"].fillna("unknown")
    df["is_generic_skill"] = (
        df["skill_label"].astype(str).str.strip().str.lower().isin(GENERIC_SKILL_LABELS)
    )
    return df


def aggregate_skill_sector_year(profile_skill_sector):
    analysis_columns = [
        "profile_uid",
        "year",
        "sector_proxy",
        "skill_uri",
        "skill_label",
        "is_generic_skill",
    ]
    df = profile_skill_sector[analysis_columns].copy()

    unique_mentions = df.drop_duplicates(
        ["profile_uid", "year", "sector_proxy", "skill_uri"]
    )

    profiles_by_sector_year = (
        df.drop_duplicates(["profile_uid", "year", "sector_proxy"])
        .groupby(["year", "sector_proxy"], as_index=False)
        .agg(profiles_in_sector_year=("profile_uid", "nunique"))
    )

    skill_counts = (
        unique_mentions.groupby(
            ["year", "sector_proxy", "skill_uri", "skill_label", "is_generic_skill"],
            as_index=False,
        )
        .agg(mentions=("profile_uid", "nunique"))
    )

    result = skill_counts.merge(
        profiles_by_sector_year,
        on=["year", "sector_proxy"],
        how="left",
    )
    result["adoption_rate"] = result["mentions"] / result["profiles_in_sector_year"]
    return result.sort_values(["skill_label", "sector_proxy", "year"])

