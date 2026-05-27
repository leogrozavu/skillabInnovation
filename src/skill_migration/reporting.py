import html
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .diffusion import build_diffusion_events, detect_sector_entries


PALETTE = [
    "#0f766e",
    "#2563eb",
    "#b45309",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0891b2",
    "#c2410c",
]


def generate_jury_artifacts(
    output_dir,
    profiles,
    profile_skill_sector,
    skill_sector_year,
    entries,
    diffusion_events,
    leaderboard,
    predictions,
    job_demand=None,
    params=None,
):
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "jury_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    params = params or {}

    metrics = build_metrics(
        profiles,
        profile_skill_sector,
        skill_sector_year,
        entries,
        diffusion_events,
        leaderboard,
        predictions,
        job_demand,
    )
    write_json(artifacts_dir / "metrics_summary.json", metrics)

    threshold_sensitivity = build_threshold_sensitivity(skill_sector_year)
    threshold_sensitivity.to_csv(artifacts_dir / "threshold_sensitivity.csv", index=False)

    bridge_skills = build_bridge_skills(skill_sector_year)
    bridge_skills.to_csv(artifacts_dir / "bridge_skills.csv", index=False)

    sector_convergence = build_sector_convergence(skill_sector_year)
    sector_convergence.to_csv(artifacts_dir / "sector_convergence.csv", index=False)

    write_bar_chart(
        artifacts_dir / "top_diffusion_or_entries.svg",
        diffusion_or_entry_chart_data(leaderboard, entries),
        "Top diffusion signals",
        "score",
    )
    write_bar_chart(
        artifacts_dir / "next_sector_radar.svg",
        prediction_chart_data(predictions),
        "Next-sector opportunity radar",
        "prediction score",
    )
    write_heatmap(
        artifacts_dir / "sector_skill_heatmap.svg",
        build_sector_skill_heatmap(skill_sector_year),
        "Recent adoption heatmap",
    )

    report = build_report_markdown(
        metrics,
        threshold_sensitivity,
        bridge_skills,
        sector_convergence,
        leaderboard,
        entries,
        predictions,
        params,
    )
    (artifacts_dir / "report.md").write_text(report, encoding="utf-8")

    pitch = build_pitch_markdown(metrics, leaderboard, bridge_skills, predictions)
    (artifacts_dir / "pitch_outline.md").write_text(pitch, encoding="utf-8")
    (artifacts_dir / "pitch_deck.html").write_text(
        build_pitch_deck_html(metrics, leaderboard, bridge_skills, predictions),
        encoding="utf-8",
    )
    (artifacts_dir / "submission_manifest.md").write_text(
        build_submission_manifest(metrics),
        encoding="utf-8",
    )

    return artifacts_dir


def build_metrics(
    profiles,
    profile_skill_sector,
    skill_sector_year,
    entries,
    diffusion_events,
    leaderboard,
    predictions,
    job_demand,
):
    year_min = int(skill_sector_year["year"].min()) if not skill_sector_year.empty else None
    year_max = int(skill_sector_year["year"].max()) if not skill_sector_year.empty else None
    return {
        "profiles": int(profiles["profile_uid"].nunique()) if "profile_uid" in profiles else len(profiles),
        "profile_skill_sector_rows": int(len(profile_skill_sector)),
        "unique_skills": int(skill_sector_year["skill_uri"].nunique()) if not skill_sector_year.empty else 0,
        "sectors": int(skill_sector_year["sector_proxy"].nunique()) if not skill_sector_year.empty else 0,
        "year_min": year_min,
        "year_max": year_max,
        "sector_entries": int(len(entries)),
        "diffusion_events": int(len(diffusion_events)),
        "diffusing_skills": int(leaderboard["skill_uri"].nunique()) if not leaderboard.empty else 0,
        "predicted_skill_sector_pairs": int(len(predictions)),
        "job_demand_signals": int(len(job_demand)) if job_demand is not None else 0,
        "generic_skill_entries": int(entries["is_generic_skill"].sum())
        if "is_generic_skill" in entries
        else 0,
        "generic_skill_predictions": int(predictions["is_generic_skill"].sum())
        if "is_generic_skill" in predictions
        else 0,
    }


def build_threshold_sensitivity(skill_sector_year):
    rows = []
    for min_mentions in [1, 2, 3, 5]:
        for min_adoption_rate in [0.001, 0.0025, 0.005, 0.01]:
            for require_stability in [False, True]:
                entries = detect_sector_entries(
                    skill_sector_year,
                    min_mentions=min_mentions,
                    min_adoption_rate=min_adoption_rate,
                    require_stability=require_stability,
                )
                events = build_diffusion_events(entries)
                rows.append(
                    {
                        "min_mentions": min_mentions,
                        "min_adoption_rate": min_adoption_rate,
                        "require_stability": require_stability,
                        "sector_entries": len(entries),
                        "diffusion_events": len(events),
                        "skills_with_entries": entries["skill_uri"].nunique()
                        if not entries.empty
                        else 0,
                        "diffusing_skills": events["skill_uri"].nunique()
                        if not events.empty
                        else 0,
                    }
                )
    return pd.DataFrame(rows)


def build_bridge_skills(skill_sector_year):
    if skill_sector_year.empty:
        return pd.DataFrame()

    return (
        skill_sector_year.groupby(["skill_uri", "skill_label"], as_index=False)
        .agg(
            sectors_observed=("sector_proxy", "nunique"),
            years_observed=("year", "nunique"),
            first_year=("year", "min"),
            latest_year=("year", "max"),
            total_mentions=("mentions", "sum"),
            max_adoption_rate=("adoption_rate", "max"),
        )
        .sort_values(
            ["sectors_observed", "total_mentions", "max_adoption_rate"],
            ascending=[False, False, False],
        )
    )


def build_sector_convergence(skill_sector_year, top_n=40):
    if skill_sector_year.empty:
        return pd.DataFrame()

    pivot = skill_sector_year.pivot_table(
        index="sector_proxy",
        columns="skill_uri",
        values="adoption_rate",
        aggfunc="mean",
        fill_value=0,
    )
    labels = (
        skill_sector_year.drop_duplicates("skill_uri")
        .set_index("skill_uri")["skill_label"]
        .to_dict()
    )
    rows = []
    for left, right in combinations(pivot.index, 2):
        left_vec = pivot.loc[left].to_numpy(dtype=float)
        right_vec = pivot.loc[right].to_numpy(dtype=float)
        denominator = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
        if np.isclose(denominator, 0):
            similarity = 0.0
        else:
            similarity = float(np.dot(left_vec, right_vec) / denominator)
        shared = (pivot.loc[left] * pivot.loc[right]).sort_values(ascending=False).head(3)
        rows.append(
            {
                "sector_a": left,
                "sector_b": right,
                "cosine_similarity": similarity,
                "shared_signal_skills": "; ".join(
                    labels.get(skill_uri, skill_uri)
                    for skill_uri, value in shared.items()
                    if value > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).head(top_n)


def build_sector_skill_heatmap(skill_sector_year, max_skills=12, max_sectors=8):
    if skill_sector_year.empty:
        return pd.DataFrame()

    max_year = int(skill_sector_year["year"].max())
    recent = skill_sector_year[skill_sector_year["year"] >= max_year - 2].copy()
    top_skills = (
        recent.groupby(["skill_uri", "skill_label"], as_index=False)
        .agg(total_mentions=("mentions", "sum"))
        .sort_values("total_mentions", ascending=False)
        .head(max_skills)
    )
    top_sectors = (
        recent.groupby("sector_proxy", as_index=False)
        .agg(total_mentions=("mentions", "sum"))
        .sort_values("total_mentions", ascending=False)
        .head(max_sectors)["sector_proxy"]
        .tolist()
    )
    heat = recent[
        recent["skill_uri"].isin(top_skills["skill_uri"])
        & recent["sector_proxy"].isin(top_sectors)
    ].pivot_table(
        index="sector_proxy",
        columns="skill_label",
        values="adoption_rate",
        aggfunc="mean",
        fill_value=0,
    )
    return heat.reindex(top_sectors)


def diffusion_or_entry_chart_data(leaderboard, entries):
    if not leaderboard.empty:
        data = leaderboard.head(10).copy()
        data["label"] = data["skill_label"] + " -> " + data["origin_sector"]
        data["value"] = data["diffusion_score"]
        return data[["label", "value"]]

    data = entries.sort_values("entry_adoption_rate", ascending=False).head(10).copy()
    if data.empty:
        return pd.DataFrame(columns=["label", "value"])
    data["label"] = data["skill_label"] + " in " + data["entered_sector"]
    data["value"] = data["entry_adoption_rate"]
    return data[["label", "value"]]


def prediction_chart_data(predictions):
    if predictions.empty:
        return pd.DataFrame(columns=["label", "value"])
    sort_columns = ["recent_mentions", "prediction_score"]
    data = predictions.sort_values(sort_columns, ascending=[False, False]).head(10).copy()
    data["label"] = data["skill_label"] + " -> " + data["candidate_sector"]
    data["value"] = data["prediction_score"]
    return data[["label", "value"]]


def build_report_markdown(
    metrics,
    threshold_sensitivity,
    bridge_skills,
    sector_convergence,
    leaderboard,
    entries,
    predictions,
    params,
):
    top_diffusion = top_rows(
        leaderboard,
        ["skill_label", "origin_sector", "sectors_reached", "diffusion_score"],
        fallback="No strict diffusion events were detected at the selected threshold.",
    )
    top_entries = top_rows(
        entries.sort_values(["entry_mentions", "entry_adoption_rate"], ascending=[False, False])
        if not entries.empty
        else entries,
        [
            "skill_label",
            "entered_sector",
            "entered_year",
            "entry_mentions",
            "entry_adoption_rate",
            "entry_profiles_in_sector_year",
        ],
    )
    strong_entries_df = strong_evidence_entries(entries, params)
    weak_entries_df = weak_evidence_entries(entries, params)
    generic_entries_df = (
        entries[entries["is_generic_skill"]].copy()
        if "is_generic_skill" in entries and not entries.empty
        else pd.DataFrame()
    )
    top_strong_entries = top_rows(
        strong_entries_df.sort_values(["entry_mentions", "entry_adoption_rate"], ascending=[False, False])
        if not strong_entries_df.empty
        else strong_entries_df,
        ["skill_label", "entered_sector", "entered_year", "entry_mentions", "entry_adoption_rate"],
        fallback="No strong-evidence entries were detected at the selected threshold.",
    )
    top_weak_entries = top_rows(
        weak_entries_df.sort_values(["entry_mentions", "entry_adoption_rate"], ascending=[True, False])
        if not weak_entries_df.empty
        else weak_entries_df,
        ["skill_label", "entered_sector", "entered_year", "entry_mentions", "entry_adoption_rate"],
        fallback="No weak-evidence entries were detected at the selected threshold.",
    )
    top_generic_entries = top_rows(
        generic_entries_df.sort_values(["entry_mentions", "entry_adoption_rate"], ascending=[False, False])
        if not generic_entries_df.empty
        else generic_entries_df,
        ["skill_label", "entered_sector", "entered_year", "entry_mentions", "entry_adoption_rate"],
        fallback="No configured generic skills appeared in the selected entries.",
    )
    top_bridge = top_rows(
        bridge_skills,
        ["skill_label", "sectors_observed", "years_observed", "total_mentions"],
    )
    top_predictions = top_rows(
        predictions.sort_values(["recent_mentions", "prediction_score"], ascending=[False, False])
        if not predictions.empty
        else predictions,
        [
            "skill_label",
            "candidate_sector",
            "recent_mentions",
            "prediction_score",
            "confidence_band",
            "reason",
        ],
    )
    top_convergence = top_rows(
        sector_convergence,
        ["sector_a", "sector_b", "cosine_similarity", "shared_signal_skills"],
    )
    sensitivity = top_rows(
        threshold_sensitivity.sort_values(
            ["require_stability", "min_mentions", "min_adoption_rate"]
        ),
        [
            "min_mentions",
            "min_adoption_rate",
            "require_stability",
            "sector_entries",
            "diffusion_events",
            "diffusing_skills",
        ],
        limit=12,
    )
    weak_entries_count = len(weak_entries_df)
    demand_note = demand_usage_note(metrics, predictions)
    example_interpretation = build_example_interpretation(entries, predictions)

    return f"""# SKILLAB Skill Migration Radar

## Executive Summary

This package turns ESCO profile histories and optional SKILLAB Tracker demand signals into a reproducible radar for skill movement across occupation-derived sectors. The current run covers **{metrics["profiles"]:,} profiles**, **{metrics["unique_skills"]:,} ESCO skills**, **{metrics["sectors"]} sector proxies**, and years **{metrics["year_min"]}-{metrics["year_max"]}**.

At the selected threshold it detected **{metrics["sector_entries"]:,} sector entries**, **{metrics["diffusion_events"]:,} strict diffusion events**, and **{metrics["predicted_skill_sector_pairs"]:,} next-sector opportunities**. **{weak_entries_count:,} entries** are weak-evidence signals by the configured thresholds and should be inspected before being used as headline findings. {demand_note}

## Main Findings

### Strict Diffusion Signals

{top_diffusion}

### First Stable Sector Entries

{top_entries}

### Strong Evidence Entries

{top_strong_entries}

### Weak Evidence Entries

{top_weak_entries}

### Configured Generic Skills

{top_generic_entries}

### Cross-sector Bridge Skills

{top_bridge}

### Next-sector Opportunity Radar

{top_predictions}

### Sector Convergence

{top_convergence}

## How to Interpret the Outputs

- `sector_entries.csv` shows the first year a skill passed the entry thresholds in a sector. Use `entry_mentions`, `entry_adoption_rate`, and `entry_profiles_in_sector_year` together; a high rate with few mentions is weaker evidence.
- `diffusion_events.csv` treats later sector entries as migration events after the inferred origin sector. This is temporal evidence, not proof that one sector caused another to adopt the skill.
- `diffusion_leaderboard.csv` ranks skills by spread across sectors and sector entropy. Use it for prioritisation, then validate the underlying entry rows.
- `next_sector_predictions.csv` is a ranking, not a probability model. `confidence_band` is derived from the relative prediction score: low signals need manual validation, medium signals are plausible leads, and high signals are the strongest candidates in this run.

Example: {example_interpretation}

## Methodology

1. Load profile parquet files and ESCO skill/occupation mappings.
2. Explode each profile into profile-skill-occupation rows.
3. Map occupations to sector proxies from ISCO prefixes.
4. Calculate annual skill adoption rate per sector: unique profiles mentioning a skill divided by profiles in that sector-year.
5. Detect first sector entry when mentions, adoption rate, and minimum sector-year profile count pass configured thresholds.
6. Build diffusion events when a skill enters another sector after its origin sector.
7. Rank next-sector opportunities using recent profile growth, recent adoption, global growth, optional job demand, demand growth, and sector similarity.

## Threshold Sensitivity

{sensitivity}

## Limitations

- Sector labels are occupation-derived proxies, not official industry classifications.
- Sparse skills can look dramatic when a sector-year has few profiles, so thresholds and stability checks matter.
- Diffusion direction is inferred from first observed entry year, not from causal movement of individual people.
- Job demand signals are included only when Tracker credentials/token are provided.

## Reproduction

```bash
python linkedin_profile.py --min-mentions {params.get("min_mentions", 2)} --min-adoption-rate {params.get("min_adoption_rate", 0.005)} --min-sector-year-profiles {params.get("min_sector_year_profiles", 50)} --require-stability
```

Generated jury artefacts are in `data/processed/jury_artifacts/`.
"""


def strong_evidence_entries(entries, params):
    if entries is None or entries.empty:
        return pd.DataFrame()
    min_mentions = int(params.get("min_mentions", 2))
    generic = (
        entries["is_generic_skill"]
        if "is_generic_skill" in entries
        else pd.Series(False, index=entries.index)
    )
    return entries[
        (entries["entry_mentions"] >= max(min_mentions * 2, min_mentions + 2))
        & (~generic)
    ].copy()


def weak_evidence_entries(entries, params):
    if entries is None or entries.empty:
        return pd.DataFrame()
    min_mentions = int(params.get("min_mentions", 2))
    weak = entries["entry_mentions"] <= min_mentions
    if "is_generic_skill" in entries:
        weak = weak | entries["is_generic_skill"]
    return entries[weak].copy()


def demand_usage_note(metrics, predictions):
    if metrics.get("job_demand_signals", 0) == 0:
        return "No Skillab Tracker demand signals were loaded, so demand score columns are zero and predictions rely on profile evidence plus sector similarity."
    if predictions is not None and not predictions.empty and "job_demand_score" in predictions:
        nonzero = int((predictions["job_demand_score"] > 0).sum())
        return f"Skillab Tracker demand contributed to {nonzero:,} prediction rows."
    return "Skillab Tracker demand signals were loaded for this run."


def build_example_interpretation(entries, predictions):
    if entries is not None and not entries.empty:
        row = entries.sort_values(["entry_mentions", "entry_adoption_rate"], ascending=[False, False]).iloc[0]
        return (
            f"`{row['skill_label']}` first passed the entry threshold in "
            f"`{row['entered_sector']}` in {int(row['entered_year'])}, with "
            f"{int(row['entry_mentions'])} profile mentions and adoption rate "
            f"{float(row['entry_adoption_rate']):.4f}."
        )
    if predictions is not None and not predictions.empty:
        row = predictions.sort_values("prediction_score", ascending=False).iloc[0]
        return (
            f"`{row['skill_label']}` is ranked for `{row['candidate_sector']}` with "
            f"score {float(row['prediction_score']):.3f} and confidence "
            f"`{row.get('confidence_band', 'unknown')}`."
        )
    return "No example row is available because the current run produced no entries or predictions."


def build_pitch_markdown(metrics, leaderboard, bridge_skills, predictions):
    best_signal = "No strict diffusion event at the conservative threshold."
    if not leaderboard.empty:
        row = leaderboard.iloc[0]
        best_signal = (
            f"{row['skill_label']} originated in {row['origin_sector']} and reached "
            f"{int(row['sectors_reached'])} additional sector(s)."
        )
    best_prediction = "No prediction available."
    if not predictions.empty:
        row = predictions.sort_values(
            ["recent_mentions", "prediction_score"],
            ascending=[False, False],
        ).iloc[0]
        best_prediction = (
            f"{row['skill_label']} -> {row['candidate_sector']} "
            f"(score {row['prediction_score']:.3f}): {row['reason']}."
        )
    best_bridge = "No bridge skill available."
    if not bridge_skills.empty:
        row = bridge_skills.iloc[0]
        best_bridge = (
            f"{row['skill_label']} appears across {int(row['sectors_observed'])} sectors "
            f"over {int(row['years_observed'])} observed years."
        )

    return f"""# Pitch Outline

## Slide 1 - Problem
Skill needs move across occupations before they show up as formal labour-market shifts. Teams and educators need an early warning system, not only historical counts.

## Slide 2 - Data Fusion
We combine ESCO skills, ESCO/ISCO occupation structure, profile histories, and optional SKILLAB Tracker job-demand signals.

## Slide 3 - Method
For each year and sector proxy, we compute profile-level skill adoption rates, detect first stable entries, infer diffusion events, and score likely next sectors.

## Slide 4 - Evidence
Current run: {metrics["profiles"]:,} profiles, {metrics["unique_skills"]:,} skills, {metrics["sectors"]} sectors, {metrics["year_min"]}-{metrics["year_max"]}. {best_signal}

## Slide 5 - Radar
{best_prediction} Bridge-skill evidence: {best_bridge}

## Slide 6 - Why It Matters
The output is reproducible, threshold-audited, and directly usable by curriculum designers, workforce analysts, and regional innovation teams.
"""


def build_pitch_deck_html(metrics, leaderboard, bridge_skills, predictions):
    best_signal = "Conservative thresholding still finds cross-sector skill movement."
    if not leaderboard.empty:
        row = leaderboard.iloc[0]
        best_signal = (
            f"{escape(row['skill_label'])} moved from {escape(row['origin_sector'])} "
            f"to {int(row['sectors_reached'])} additional sector(s)."
        )

    best_prediction = "The radar ranks likely next sectors from profile growth, demand, and sector similarity."
    if not predictions.empty:
        row = predictions.sort_values(
            ["recent_mentions", "prediction_score"],
            ascending=[False, False],
        ).iloc[0]
        best_prediction = (
            f"{escape(row['skill_label'])} -> {escape(row['candidate_sector'])}, "
            f"score {float(row['prediction_score']):.3f}."
        )

    best_bridge = "Bridge skills reveal convergence before formal sector labels catch up."
    if not bridge_skills.empty:
        row = bridge_skills.iloc[0]
        best_bridge = (
            f"{escape(row['skill_label'])} appears across "
            f"{int(row['sectors_observed'])} sectors."
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SKILLAB Skill Migration Radar</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --accent: #0f766e;
      --blue: #2563eb;
      --gold: #b45309;
      --rose: #be123c;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: #f8fafc;
    }}
    section {{
      min-height: 100vh;
      box-sizing: border-box;
      padding: 7vh 7vw;
      display: grid;
      align-content: center;
      gap: 28px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
      line-height: 1.05;
    }}
    h1 {{ font-size: 64px; max-width: 980px; }}
    h2 {{ font-size: 48px; }}
    p {{ max-width: 860px; font-size: 24px; line-height: 1.35; color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 16px;
      max-width: 1000px;
    }}
    .metric {{
      border-top: 4px solid var(--accent);
      padding-top: 12px;
    }}
    .metric strong {{ display: block; font-size: 34px; }}
    .metric span {{ color: var(--muted); font-size: 15px; }}
    img {{
      width: min(100%, 1100px);
      height: auto;
      border: 1px solid var(--line);
      background: white;
    }}
    .accent {{ color: var(--accent); }}
    .two {{
      grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
      align-items: center;
    }}
    @media (max-width: 800px) {{
      h1 {{ font-size: 42px; }}
      h2 {{ font-size: 34px; }}
      p {{ font-size: 19px; }}
      section, .two {{ display: block; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <section>
    <h1>SKILLAB <span class="accent">Skill Migration Radar</span></h1>
    <p>Early-warning analytics for how ESCO skills move across occupation-derived sectors over time.</p>
    <div class="metrics">
      <div class="metric"><strong>{metrics["profiles"]:,}</strong><span>profiles</span></div>
      <div class="metric"><strong>{metrics["unique_skills"]:,}</strong><span>ESCO skills</span></div>
      <div class="metric"><strong>{metrics["sectors"]}</strong><span>sector proxies</span></div>
      <div class="metric"><strong>{metrics["year_min"]}-{metrics["year_max"]}</strong><span>time span</span></div>
    </div>
  </section>
  <section class="two">
    <div>
      <h2>What Changed</h2>
      <p>{best_signal}</p>
      <p>The method detects first sector entries, then turns later entries into migration events.</p>
    </div>
    <img src="top_diffusion_or_entries.svg" alt="Top diffusion signals">
  </section>
  <section class="two">
    <div>
      <h2>Opportunity Radar</h2>
      <p>{best_prediction}</p>
      <p>Scores combine adoption growth, recent support, optional demand, and similarity to sectors that already adopted the skill.</p>
    </div>
    <img src="next_sector_radar.svg" alt="Next-sector opportunity radar">
  </section>
  <section class="two">
    <div>
      <h2>Sector Convergence</h2>
      <p>{best_bridge}</p>
      <p>The heatmap shows recent adoption intensity for high-signal skills across the strongest sectors.</p>
    </div>
    <img src="sector_skill_heatmap.svg" alt="Recent adoption heatmap">
  </section>
  <section>
    <h2>Why This Is Useful</h2>
    <p>The output is reproducible, threshold-audited, and packaged for curriculum designers, workforce analysts, and regional innovation teams.</p>
    <div class="metrics">
      <div class="metric"><strong>{metrics["sector_entries"]:,}</strong><span>sector entries</span></div>
      <div class="metric"><strong>{metrics["diffusion_events"]:,}</strong><span>diffusion events</span></div>
      <div class="metric"><strong>{metrics["predicted_skill_sector_pairs"]:,}</strong><span>opportunities</span></div>
      <div class="metric"><strong>{metrics["job_demand_signals"]:,}</strong><span>Tracker demand signals</span></div>
    </div>
  </section>
</body>
</html>
"""


def build_submission_manifest(metrics):
    return f"""# Submission Manifest

## Code

- `linkedin_profile.py`
- `src/skill_migration/`
- `requirements.txt`
- `README.md`

## Results

- `data/processed/profile_skill_sector.parquet`
- `data/processed/skill_sector_year.parquet`
- `data/processed/sector_entries.csv`
- `data/processed/diffusion_events.csv`
- `data/processed/diffusion_leaderboard.csv`
- `data/processed/next_sector_predictions.csv`
- `data/processed/jury_artifacts/*.csv`
- `data/processed/jury_artifacts/*.svg`
- `data/processed/jury_artifacts/metrics_summary.json`

## Report

- `data/processed/jury_artifacts/report.md`

## Pitch

- `data/processed/jury_artifacts/pitch_outline.md`
- `data/processed/jury_artifacts/pitch_deck.html`

## Run Summary

- Profiles: {metrics["profiles"]:,}
- ESCO skills: {metrics["unique_skills"]:,}
- Sector proxies: {metrics["sectors"]}
- Years: {metrics["year_min"]}-{metrics["year_max"]}
- Sector entries: {metrics["sector_entries"]:,}
- Diffusion events: {metrics["diffusion_events"]:,}
- Next-sector opportunities: {metrics["predicted_skill_sector_pairs"]:,}
"""


def top_rows(df, columns, limit=8, fallback="No rows available."):
    if df is None or df.empty:
        return fallback
    available = [column for column in columns if column in df.columns]
    if not available:
        return fallback
    return markdown_table(df[available].head(limit))


def markdown_table(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_bar_chart(path, data, title, x_label):
    width = 1100
    row_height = 42
    margin_left = 360
    margin_top = 70
    chart_width = 650
    height = max(240, margin_top + row_height * max(len(data), 1) + 50)
    max_value = float(data["value"].max()) if not data.empty and data["value"].max() > 0 else 1.0
    rows = []
    for index, row in enumerate(data.itertuples(index=False)):
        y = margin_top + index * row_height
        value = float(row.value)
        bar_width = max(2, int(chart_width * value / max_value))
        color = PALETTE[index % len(PALETTE)]
        label = truncate(str(row.label), 48)
        rows.append(
            f'<text x="24" y="{y + 20}" class="label">{escape(label)}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="26" rx="4" fill="{color}" />'
            f'<text x="{margin_left + bar_width + 8}" y="{y + 19}" class="value">{value:.3f}</text>'
        )
    if not rows:
        rows.append(f'<text x="24" y="{margin_top}" class="label">No data available</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 26px Arial, sans-serif; fill: #111827; }}
    .axis {{ font: 13px Arial, sans-serif; fill: #4b5563; }}
    .label {{ font: 14px Arial, sans-serif; fill: #111827; }}
    .value {{ font: 13px Arial, sans-serif; fill: #374151; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="36" class="title">{escape(title)}</text>
  <text x="{margin_left}" y="58" class="axis">{escape(x_label)}</text>
  {''.join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_heatmap(path, heat, title):
    width = 1250
    cell_w = 76
    cell_h = 34
    margin_left = 250
    margin_top = 150
    rows_count = len(heat.index) if not heat.empty else 1
    cols_count = len(heat.columns) if not heat.empty else 1
    height = margin_top + rows_count * cell_h + 60
    max_value = float(heat.to_numpy().max()) if not heat.empty and heat.to_numpy().max() > 0 else 1.0

    cells = []
    if heat.empty:
        cells.append(f'<text x="24" y="{margin_top}" class="label">No data available</text>')
    else:
        for col_idx, column in enumerate(heat.columns):
            x = margin_left + col_idx * cell_w + 8
            cells.append(
                f'<text x="{x}" y="135" transform="rotate(-45 {x} 135)" class="small">{escape(truncate(column, 24))}</text>'
            )
        for row_idx, sector in enumerate(heat.index):
            y = margin_top + row_idx * cell_h
            cells.append(f'<text x="24" y="{y + 23}" class="label">{escape(sector)}</text>')
            for col_idx, column in enumerate(heat.columns):
                value = float(heat.loc[sector, column])
                intensity = value / max_value
                color = interpolate_color((235, 247, 245), (15, 118, 110), intensity)
                x = margin_left + col_idx * cell_w
                cells.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" fill="{color}" />'
                    f'<text x="{x + 8}" y="{y + 22}" class="cell">{value:.2f}</text>'
                )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 26px Arial, sans-serif; fill: #111827; }}
    .label {{ font: 14px Arial, sans-serif; fill: #111827; }}
    .small {{ font: 12px Arial, sans-serif; fill: #374151; }}
    .cell {{ font: 11px Arial, sans-serif; fill: #111827; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="36" class="title">{escape(title)}</text>
  {''.join(cells)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def interpolate_color(start, end, intensity):
    intensity = max(0.0, min(1.0, float(intensity)))
    rgb = [round(s + (e - s) * intensity) for s, e in zip(start, end)]
    return "#" + "".join(f"{value:02x}" for value in rgb)


def truncate(value, max_len):
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def escape(value):
    return html.escape(str(value), quote=True)
