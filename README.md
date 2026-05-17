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

## Outputs

- `profile_skill_sector.parquet`: normalized profile-skill-occupation-sector rows.
- `skill_sector_year.parquet`: adoption rate by year, sector, and skill.
- `sector_entries.csv`: first stable entry of each skill into each sector.
- `diffusion_events.csv`: migration events after origin detection.
- `diffusion_leaderboard.csv`: ranked skills by diffusion score.
- `next_sector_predictions.csv`: simple next-sector predictions.
