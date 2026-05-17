import os

import numpy as np
import pandas as pd

from .api_client import SkillabClient


ID_KEYS = (
    "skill_id",
    "skill_uri",
    "id",
    "uri",
    "conceptUri",
    "concept_uri",
)
LABEL_KEYS = (
    "skill_label",
    "label",
    "preferredLabel",
    "preferred_label",
    "name",
    "skill",
)
COUNT_KEYS = (
    "count",
    "appearances",
    "frequency",
    "total",
    "value",
    "n",
    "num_appearances",
)


def load_skillab_client(username=None, password=None, token=None, timeout=60):
    token = token or os.getenv("SKILLAB_TOKEN") or os.getenv("TRACKER_TOKEN")
    username = username or os.getenv("SKILLAB_USERNAME") or os.getenv("TRACKER_USERNAME")
    password = password or os.getenv("SKILLAB_PASSWORD") or os.getenv("TRACKER_PASSWORD")

    client = SkillabClient(token=token, timeout=timeout)
    if not client.token and username and password:
        client.login(username, password)
    return client


def fetch_job_demand_signals(
    username=None,
    password=None,
    token=None,
    limit=500,
    source=None,
    current_from_date=None,
    current_to_date=None,
    baseline_from_date=None,
    baseline_to_date=None,
    timeout=60,
):
    client = load_skillab_client(
        username=username,
        password=password,
        token=token,
        timeout=timeout,
    )
    if not client.token:
        raise ValueError(
            "Skillab job demand needs SKILLAB_TOKEN or SKILLAB_USERNAME/SKILLAB_PASSWORD."
        )

    current_rows = client.get_job_skill_demand(
        limit=limit,
        source=source,
        from_date=current_from_date,
        to_date=current_to_date,
    )
    current = normalize_job_demand_rows(current_rows, count_column="job_demand_count")

    if baseline_from_date or baseline_to_date:
        baseline_rows = client.get_job_skill_demand(
            limit=limit,
            source=source,
            from_date=baseline_from_date,
            to_date=baseline_to_date,
        )
        baseline = normalize_job_demand_rows(
            baseline_rows,
            count_column="baseline_job_demand_count",
        )
        current = current.merge(
            baseline[["skill_uri", "baseline_job_demand_count"]],
            on="skill_uri",
            how="left",
        )
    else:
        current["baseline_job_demand_count"] = 0

    current["baseline_job_demand_count"] = current["baseline_job_demand_count"].fillna(0)
    current["job_demand_growth"] = (
        current["job_demand_count"] - current["baseline_job_demand_count"]
    )
    current["job_demand_score"] = min_max_score(current["job_demand_count"])
    current["job_demand_growth_score"] = min_max_score(
        current["job_demand_growth"].clip(lower=0)
    )
    current["job_demand_rank"] = current["job_demand_count"].rank(
        method="dense",
        ascending=False,
    )
    return current.sort_values("job_demand_count", ascending=False)


def normalize_job_demand_rows(rows, count_column):
    items = rows.get("items", rows) if isinstance(rows, dict) else rows
    normalized = []

    for row in items or []:
        if not isinstance(row, dict):
            continue

        skill_uri = first_present(row, ID_KEYS)
        label = first_present(row, LABEL_KEYS)
        count = first_present(row, COUNT_KEYS)

        nested_skill = row.get("skill")
        if isinstance(nested_skill, dict):
            skill_uri = skill_uri or first_present(nested_skill, ID_KEYS)
            label = label or first_present(nested_skill, LABEL_KEYS)

        if not skill_uri:
            continue

        normalized.append(
            {
                "skill_uri": str(skill_uri),
                "demand_skill_label": label,
                count_column: safe_float(count),
            }
        )

    df = pd.DataFrame(normalized)
    if df.empty:
        return pd.DataFrame(
            columns=["skill_uri", "demand_skill_label", count_column]
        )

    return (
        df.groupby(["skill_uri", "demand_skill_label"], dropna=False, as_index=False)
        .agg({count_column: "sum"})
        .sort_values(count_column, ascending=False)
    )


def first_present(row, keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def min_max_score(series):
    series = pd.Series(series).fillna(0).astype(float)
    minimum = series.min()
    maximum = series.max()
    if np.isclose(maximum, minimum):
        return pd.Series(np.where(series > 0, 1.0, 0.0), index=series.index)
    return (series - minimum) / (maximum - minimum)
