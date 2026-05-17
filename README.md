# Skillab Innovation

Prototype for analyzing skill diffusion across occupations, sectors, and time using ESCO profile data and Skillab Tracker signals.

## Run

Create/activate the virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the local Python pipeline:

```bash
python linkedin_profile.py
```

By default it reads:

```text
~/Downloads/output_profiles
~/Downloads/mapping_of_ESCO_skills.xlsx
~/Downloads/mapping_of_ESCO_occupations.xlsx
```

It writes processed outputs to:

```text
data/processed/
```

Useful options:

```bash
python linkedin_profile.py --limit-files 10
python linkedin_profile.py --limit-files 0
python linkedin_profile.py --min-mentions 3 --min-adoption-rate 0.002
python linkedin_profile.py --require-stability
```

To blend Skillab job demand into predictions, provide a bearer token or credentials:

```bash
export TRACKER_API="https://skillab-tracker.csd.auth.gr/api"
export TRACKER_USERNAME="event_public"
export TRACKER_PASSWORD="PublicOnly2026"
python linkedin_profile.py --use-skillab-demand --limit-files 5
```

You can also use `SKILLAB_TOKEN` / `SKILLAB_USERNAME` / `SKILLAB_PASSWORD`.

With a baseline/current demand comparison:

```bash
python linkedin_profile.py \
  --use-skillab-demand \
  --demand-baseline-from-date 2023-01-01 \
  --demand-baseline-to-date 2023-12-31 \
  --demand-current-from-date 2024-01-01 \
  --demand-current-to-date 2024-12-31
```

## Outputs

- `profile_skill_sector.parquet`: normalized profile-skill-occupation-sector rows.
- `skill_sector_year.parquet`: adoption rate by year, sector, and skill.
- `sector_entries.csv`: first stable entry of each skill into each sector.
- `diffusion_events.csv`: migration events after origin detection.
- `diffusion_leaderboard.csv`: ranked skills by diffusion score.
- `next_sector_predictions.csv`: simple next-sector predictions.
- `job_demand_signals.csv`: Skillab job demand signal per skill, when `--use-skillab-demand` is enabled.

`next_sector_predictions.csv` includes separate score components:

- `profile_growth_score`
- `recent_adoption_score`
- `global_profile_growth_score`
- `job_demand_score`
- `job_demand_growth_score`
- `sector_similarity_score`
- `prediction_score`
- `reason`
