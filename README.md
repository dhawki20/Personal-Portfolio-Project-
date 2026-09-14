# Nick Hawkins - Personal Portfolio Project

This repository contains my DTSC 2301 professional portfolio and first end-to-end exploratory data analysis project.

## Research question

Did NFL teams that made a higher percentage of their field goals also win more games during the 2025 regular season?

## Main result

Across all 32 teams, the relationship was weak and not statistically significant (Pearson r = 0.14, p = 0.44, R-squared = 0.02). The result is an association and does not establish causation.

## Repository guide

- `index.html`, `about.html`, `project.html`: GitHub Pages portfolio
- `notebooks/nfl_field_goal_analysis.ipynb`: documented Jupyter analysis
- `scripts/build_analysis.py`: reproducible data-processing and chart script
- `data/raw/`: cached official nflverse release data
- `data/processed/`: cleaned one-row-per-team analysis data
- `assets/`: final charts
- `resume/`: downloadable resume files

## Run the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_analysis.py
jupyter notebook notebooks/nfl_field_goal_analysis.ipynb
```

Data are accessed using [nflreadpy](https://nflreadpy.nflverse.com/api/load_functions/) and official [nflverse-data releases](https://github.com/nflverse/nflverse-data/releases).

## GitHub Pages

In the repository settings, open **Pages**, choose **Deploy from a branch**, select the `main` branch and `/ (root)` folder, then save.
