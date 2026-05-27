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


def predict_next_sectors(
    skill_sector_year,
    diffusion_events,
    job_demand=None,
    lookback_years=3,
    top_n=5,
):
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
    predictions["evidence_support_score"] = min_max_score(np.log1p(predictions["recent_mentions"]))
    predictions["profile_growth_score"] = min_max_score(
        predictions["growth_rate"].clip(lower=0)
    )
    predictions["recent_adoption_score"] = min_max_score(
        predictions["recent_adoption_rate"]
    )
    predictions["global_profile_growth_score"] = min_max_score(
        predictions["global_growth"].clip(lower=0)
    )
    predictions = add_sector_similarity_score(predictions, recent, diffusion_events)
    predictions = add_job_demand_scores(predictions, job_demand)
    predictions["prediction_score"] = (
        0.20 * predictions["profile_growth_score"]
        + 0.10 * predictions["recent_adoption_score"]
        + 0.15 * predictions["global_profile_growth_score"]
        + 0.15 * predictions["evidence_support_score"]
        + 0.25 * predictions["job_demand_score"]
        + 0.10 * predictions["job_demand_growth_score"]
        + 0.05 * predictions["sector_similarity_score"]
    )
    predictions["reason"] = predictions.apply(build_prediction_reason, axis=1)

    return (
        predictions.sort_values(["skill_label", "prediction_score"], ascending=[True, False])
        .groupby(["skill_uri", "skill_label"], as_index=False)
        .head(top_n)
        .drop(columns=["entered_sector", "already_entered"], errors="ignore")
    )


def add_job_demand_scores(predictions, job_demand):
    if job_demand is None or job_demand.empty:
        predictions["job_demand_count"] = 0.0
        predictions["baseline_job_demand_count"] = 0.0
        predictions["job_demand_growth"] = 0.0
        predictions["job_demand_score"] = 0.0
        predictions["job_demand_growth_score"] = 0.0
        return predictions

    demand_columns = [
        "skill_uri",
        "job_demand_count",
        "baseline_job_demand_count",
        "job_demand_growth",
        "job_demand_score",
        "job_demand_growth_score",
    ]
    available_columns = [column for column in demand_columns if column in job_demand.columns]
    predictions = predictions.merge(
        job_demand[available_columns].drop_duplicates("skill_uri"),
        on="skill_uri",
        how="left",
    )
    for column in demand_columns:
        if column != "skill_uri" and column not in predictions.columns:
            predictions[column] = 0.0
    fill_columns = [column for column in demand_columns if column != "skill_uri"]
    predictions[fill_columns] = predictions[fill_columns].fillna(0.0)
    return predictions


def add_sector_similarity_score(predictions, recent, entries):
    sector_vectors = (
        recent.pivot_table(
            index="sector_proxy",
            columns="skill_uri",
            values="adoption_rate",
            aggfunc="mean",
            fill_value=0,
        )
    )

    entered_by_skill = {}
    if entries is not None and not entries.empty:
        sector_column = "entered_sector" if "entered_sector" in entries.columns else "sector_proxy"
        for skill_uri, group in entries.groupby("skill_uri"):
            entered_by_skill[skill_uri] = set(group[sector_column].dropna())

    scores = []
    for row in predictions.itertuples(index=False):
        candidate = row.candidate_sector
        adopted_sectors = entered_by_skill.get(row.skill_uri, set())
        adopted_sectors = [sector for sector in adopted_sectors if sector in sector_vectors.index]

        if candidate not in sector_vectors.index or not adopted_sectors:
            scores.append(0.0)
            continue

        candidate_vector = sector_vectors.loc[candidate].to_numpy(dtype=float)
        candidate_norm = np.linalg.norm(candidate_vector)
        if np.isclose(candidate_norm, 0):
            scores.append(0.0)
            continue

        similarities = []
        for sector in adopted_sectors:
            sector_vector = sector_vectors.loc[sector].to_numpy(dtype=float)
            denominator = candidate_norm * np.linalg.norm(sector_vector)
            if np.isclose(denominator, 0):
                continue
            similarities.append(float(np.dot(candidate_vector, sector_vector) / denominator))

        scores.append(max(similarities) if similarities else 0.0)

    predictions["sector_similarity_score"] = scores
    return predictions


def build_prediction_reason(row):
    reasons = []
    if row.profile_growth_score >= 0.5:
        reasons.append("profile adoption is growing")
    if row.job_demand_score >= 0.5:
        reasons.append("high job demand")
    if row.job_demand_growth_score >= 0.5:
        reasons.append("job demand is increasing")
    if row.evidence_support_score >= 0.5:
        reasons.append("repeated recent profile evidence")
    if row.sector_similarity_score >= 0.5:
        reasons.append("similar sectors already adopted it")
    if row.recent_adoption_score >= 0.5:
        reasons.append("recent adoption is visible")
    return "; ".join(reasons) if reasons else "weak early signal"


def min_max_score(series):
    series = pd.Series(series).fillna(0).astype(float)
    minimum = series.min()
    maximum = series.max()
    if np.isclose(maximum, minimum):
        return pd.Series(np.where(series > 0, 1.0, 0.0), index=series.index)
    return (series - minimum) / (maximum - minimum)
