import numpy as np
import pandas as pd


def detect_sector_entries(
    skill_sector_year,
    min_mentions=3,
    min_adoption_rate=0.005,
    require_stability=False,
):
    df = skill_sector_year.copy()
    df["passes_threshold"] = (
        (df["mentions"] >= min_mentions)
        & (df["adoption_rate"] >= min_adoption_rate)
    )

    if require_stability:
        df = df.sort_values(["skill_uri", "sector_proxy", "year"])
        df["next_passes_threshold"] = df.groupby(["skill_uri", "sector_proxy"])[
            "passes_threshold"
        ].shift(-1, fill_value=False)
        df["stable_entry"] = df["passes_threshold"] & df["next_passes_threshold"]
    else:
        df["stable_entry"] = df["passes_threshold"]

    entries = df[df["stable_entry"]].copy()
    entries = (
        entries.sort_values(["skill_uri", "sector_proxy", "year"])
        .groupby(["skill_uri", "skill_label", "sector_proxy"], as_index=False)
        .first()
    )
    entries = entries.rename(
        columns={
            "sector_proxy": "entered_sector",
            "year": "entered_year",
            "mentions": "entry_mentions",
            "adoption_rate": "entry_adoption_rate",
        }
    )
    return entries[
        [
            "skill_uri",
            "skill_label",
            "entered_sector",
            "entered_year",
            "entry_mentions",
            "entry_adoption_rate",
        ]
    ]


def build_diffusion_events(entries):
    if entries.empty:
        return pd.DataFrame(
            columns=[
                "skill_uri",
                "skill_label",
                "origin_sector",
                "origin_year",
                "entered_sector",
                "entered_year",
                "delay_years",
                "entry_mentions",
                "entry_adoption_rate",
            ]
        )

    origins = (
        entries.sort_values(
            ["skill_uri", "entered_year", "entry_adoption_rate", "entry_mentions"],
            ascending=[True, True, False, False],
        )
        .groupby(["skill_uri", "skill_label"], as_index=False)
        .first()
        .rename(
            columns={
                "entered_sector": "origin_sector",
                "entered_year": "origin_year",
            }
        )[["skill_uri", "skill_label", "origin_sector", "origin_year"]]
    )

    events = entries.merge(origins, on=["skill_uri", "skill_label"], how="left")
    events = events[events["entered_sector"] != events["origin_sector"]].copy()
    events["delay_years"] = events["entered_year"] - events["origin_year"]
    events = events[events["delay_years"] > 0]
    return events.sort_values(["skill_label", "entered_year", "entered_sector"])


def diffusion_leaderboard(diffusion_events, skill_sector_year):
    if diffusion_events.empty:
        return pd.DataFrame(
            columns=[
                "skill_uri",
                "skill_label",
                "origin_sector",
                "origin_year",
                "sectors_reached",
                "first_migration_year",
                "latest_migration_year",
                "diffusion_score",
            ]
        )

    sector_counts = (
        diffusion_events.groupby(
            ["skill_uri", "skill_label", "origin_sector", "origin_year"], as_index=False
        )
        .agg(
            sectors_reached=("entered_sector", "nunique"),
            first_migration_year=("entered_year", "min"),
            latest_migration_year=("entered_year", "max"),
            mean_entry_adoption_rate=("entry_adoption_rate", "mean"),
        )
    )

    entropy = compute_skill_sector_entropy(skill_sector_year)
    latest_entropy = (
        entropy.sort_values(["skill_uri", "year"])
        .groupby("skill_uri", as_index=False)
        .tail(1)[["skill_uri", "sector_entropy"]]
    )

    leaderboard = sector_counts.merge(latest_entropy, on="skill_uri", how="left")
    leaderboard["sector_entropy"] = leaderboard["sector_entropy"].fillna(0)
    leaderboard["diffusion_score"] = (
        leaderboard["sectors_reached"]
        * (1 + leaderboard["sector_entropy"])
        * (1 + leaderboard["mean_entry_adoption_rate"])
    )
    return leaderboard.sort_values("diffusion_score", ascending=False)


def compute_skill_sector_entropy(skill_sector_year):
    latest = skill_sector_year.copy()
    totals = (
        latest.groupby(["skill_uri", "year"], as_index=False)
        .agg(total_mentions=("mentions", "sum"))
    )
    latest = latest.merge(totals, on=["skill_uri", "year"], how="left")
    latest["p"] = latest["mentions"] / latest["total_mentions"]
    latest["entropy_part"] = -latest["p"] * np.log2(latest["p"])
    return (
        latest.groupby(["skill_uri", "skill_label", "year"], as_index=False)
        .agg(sector_entropy=("entropy_part", "sum"))
    )


def predict_next_sectors(skill_sector_year, diffusion_events, lookback_years=3, top_n=5):
    df = skill_sector_year.copy()
    if df.empty:
        return pd.DataFrame()

    max_year = int(df["year"].max())
    recent = df[df["year"] >= max_year - lookback_years + 1].copy()

    growth = []
    for (skill_uri, skill_label, sector), group in recent.groupby(
        ["skill_uri", "skill_label", "sector_proxy"]
    ):
        group = group.sort_values("year")
        first = group.iloc[0]["adoption_rate"]
        last = group.iloc[-1]["adoption_rate"]
        growth.append(
            {
                "skill_uri": skill_uri,
                "skill_label": skill_label,
                "candidate_sector": sector,
                "recent_adoption_rate": last,
                "growth_rate": last - first,
                "recent_mentions": int(group.iloc[-1]["mentions"]),
            }
        )

    predictions = pd.DataFrame(growth)
    if predictions.empty:
        return predictions

    already_entered = diffusion_events[["skill_uri", "entered_sector"]].drop_duplicates()
    already_entered["already_entered"] = True
    predictions = predictions.merge(
        already_entered,
        left_on=["skill_uri", "candidate_sector"],
        right_on=["skill_uri", "entered_sector"],
        how="left",
    )
    predictions = predictions[predictions["already_entered"].isna()].copy()

    global_growth = (
        recent.groupby(["skill_uri", "skill_label", "year"], as_index=False)
        .agg(global_mentions=("mentions", "sum"))
        .sort_values(["skill_uri", "year"])
    )
    global_growth["global_growth"] = global_growth.groupby("skill_uri")[
        "global_mentions"
    ].transform(lambda series: series.iloc[-1] - series.iloc[0])
    global_growth = global_growth.groupby(["skill_uri", "skill_label"], as_index=False).tail(1)

    predictions = predictions.merge(
        global_growth[["skill_uri", "global_growth"]],
        on="skill_uri",
        how="left",
    )
    predictions["global_growth"] = predictions["global_growth"].fillna(0)
    predictions["prediction_score"] = (
        predictions["growth_rate"].clip(lower=0) * 100
        + np.log1p(predictions["recent_mentions"])
        + np.log1p(predictions["global_growth"].clip(lower=0))
    )

    return (
        predictions.sort_values(["skill_label", "prediction_score"], ascending=[True, False])
        .groupby(["skill_uri", "skill_label"], as_index=False)
        .head(top_n)
        .drop(columns=["entered_sector", "already_entered"], errors="ignore")
    )
