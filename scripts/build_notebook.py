"""Create the submitted Jupyter notebook without requiring notebook libraries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "nfl_field_goal_analysis.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# NFL Field-Goal Accuracy and Team Winning Percentage\n\n"
       "**Research question:** What was the relationship between field-goal percentage and team winning percentage during the 2025 NFL regular season?\n\n"
       "This notebook collects team kicking statistics and schedules, creates one analysis row per NFL team, documents the cleaning process, and evaluates the relationship with visualizations and correlation. This is an exploratory observational analysis; it cannot establish causation."""),
    md("""## 1. Setup and data collection\n\n"
       "The primary collection method uses [`nflreadpy`](https://nflreadpy.nflverse.com/api/load_functions/): `load_team_stats` supplies regular-season team totals and `load_schedules` supplies final scores. The repository includes cached official nflverse CSV releases as a transparent fallback, so the work can still run if the package or network is unavailable."""),
    code("""from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom matplotlib.ticker import PercentFormatter\nfrom scipy.stats import linregress, pearsonr, spearmanr\n\nROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\nRAW = ROOT / \"data\" / \"raw\"\nASSETS = ROOT / \"assets\"\nsns.set_theme(style=\"whitegrid\", context=\"talk\")"""),
    code("""try:\n    import nflreadpy as nfl\n    team_stats = nfl.load_team_stats(2025, summary_level=\"reg\").to_pandas()\n    schedules = nfl.load_schedules(2025).to_pandas()\n    collection_method = \"Loaded live with nflreadpy\"\nexcept (ImportError, OSError, ValueError):\n    team_stats = pd.read_csv(RAW / \"stats_team_reg_2025.csv\")\n    schedules = pd.read_csv(RAW / \"games.csv\", low_memory=False)\n    collection_method = \"Loaded cached official nflverse release files\"\n\nprint(collection_method)\nprint(\"Team-stat source shape:\", team_stats.shape)\nprint(\"Schedule source shape:\", schedules.shape)"""),
    md("""## 2. Cleaning and preparation\n\n"
       "I filter to completed 2025 regular-season games, reshape each game into home-team and away-team observations, and calculate records from final scores. A tie contributes half a win to winning percentage. I then recalculate field-goal percentage from makes and attempts and perform a validated one-to-one join. I keep every team because all 32 belong to the population being described."""),
    code("""regular = schedules.loc[\n    (schedules[\"season\"] == 2025)\n    & (schedules[\"game_type\"] == \"REG\")\n    & schedules[\"home_score\"].notna()\n    & schedules[\"away_score\"].notna()\n].copy()\n\nhome = regular[[\"home_team\", \"home_score\", \"away_score\"]].rename(\n    columns={\"home_team\": \"team\", \"home_score\": \"points_for\", \"away_score\": \"points_against\"})\naway = regular[[\"away_team\", \"away_score\", \"home_score\"]].rename(\n    columns={\"away_team\": \"team\", \"away_score\": \"points_for\", \"home_score\": \"points_against\"})\nteam_games = pd.concat([home, away], ignore_index=True)\nteam_games[\"win\"] = (team_games.points_for > team_games.points_against).astype(int)\nteam_games[\"loss\"] = (team_games.points_for < team_games.points_against).astype(int)\nteam_games[\"tie\"] = (team_games.points_for == team_games.points_against).astype(int)\n\nrecords = team_games.groupby(\"team\", as_index=False).agg(\n    wins=(\"win\", \"sum\"), losses=(\"loss\", \"sum\"), ties=(\"tie\", \"sum\"), games_played=(\"team\", \"size\"))\nrecords[\"winning_percentage\"] = (records.wins + 0.5 * records.ties) / records.games_played"""),
    code("""kicking = team_stats[[\"team\", \"fg_made\", \"fg_att\", \"fg_missed\", \"fg_blocked\", \"fg_long\"]].copy()\nkicking[\"field_goal_percentage\"] = kicking.fg_made / kicking.fg_att\n\nanalysis = records.merge(kicking, on=\"team\", how=\"inner\", validate=\"one_to_one\")\nanalysis[\"record\"] = analysis.wins.astype(str) + \"-\" + analysis.losses.astype(str)\nanalysis.loc[analysis.ties > 0, \"record\"] += \"-\" + analysis.loc[analysis.ties > 0, \"ties\"].astype(str)\nanalysis = analysis.sort_values(\"field_goal_percentage\", ascending=False)\n\nassert analysis.shape[0] == 32\nassert analysis.team.is_unique\nassert analysis.isna().sum().sum() == 0\nassert (analysis.games_played == 17).all()\nprint(\"Analysis rows:\", len(analysis))\nprint(\"Missing values:\", int(analysis.isna().sum().sum()))\nanalysis.head()"""),
    md("""### Variables\n\n"
       "- **Field-goal percentage (explanatory):** team field goals made divided by team field goals attempted during the regular season.\n"
       "- **Winning percentage (outcome):** wins plus half of ties, divided by games played.\n"
       "- **Supporting variables:** attempts, makes, misses, blocks, longest make, wins, losses, ties, and games played."""),
    md("""## 3. Analysis and visualization\n\n"
       "Pearson correlation summarizes the strength of a linear relationship. I also calculate Spearman rank correlation as a robustness check because it is less dependent on a straight-line pattern."""),
    code("""pearson_r, pearson_p = pearsonr(analysis.field_goal_percentage, analysis.winning_percentage)\nspearman_rho, spearman_p = spearmanr(analysis.field_goal_percentage, analysis.winning_percentage)\nreg = linregress(analysis.field_goal_percentage, analysis.winning_percentage)\n\nprint(f\"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f}\")\nprint(f\"R-squared = {reg.rvalue**2:.3f}\")\nprint(f\"Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.3f}\")"""),
    code("""fig, ax = plt.subplots(figsize=(11, 7))\nsns.regplot(data=analysis, x=\"field_goal_percentage\", y=\"winning_percentage\", ci=None,\n            scatter_kws={\"s\": 75, \"alpha\": .82, \"color\": \"#007C91\", \"edgecolor\": \"white\"},\n            line_kws={\"color\": \"#FCA311\", \"linewidth\": 2.5}, ax=ax)\nfor row in analysis.itertuples():\n    predicted = reg.intercept + reg.slope * row.field_goal_percentage\n    if (row.field_goal_percentage in [analysis.field_goal_percentage.min(), analysis.field_goal_percentage.max()]\n            or abs(row.winning_percentage - predicted) >= .19):\n        ax.annotate(row.team, (row.field_goal_percentage, row.winning_percentage), xytext=(5, 5),\n                    textcoords=\"offset points\", fontsize=9)\nax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))\nax.set(title=\"Field-Goal Accuracy and Team Winning Percentage, 2025 NFL Regular Season\",\n       xlabel=\"Team field-goal percentage\", ylabel=\"Team winning percentage\")\nplt.tight_layout(); plt.show()"""),
    md("""The fitted line rises slightly, but teams are widely dispersed. Pearson **r = 0.14**, **p = 0.44**, and **R² = 0.02**. Field-goal percentage therefore explains only about 2% of the observed team-level variation in winning percentage, and this sample does not show a statistically significant linear relationship."""),
    code("""ranked = analysis.sort_values(\"field_goal_percentage\")\nfig, ax = plt.subplots(figsize=(11, 12))\nnorm = plt.Normalize(analysis.winning_percentage.min(), analysis.winning_percentage.max())\ncmap = plt.colormaps[\"viridis\"]\nbars = ax.barh(ranked.team, ranked.field_goal_percentage,\n               color=cmap(norm(ranked.winning_percentage)), edgecolor=\"white\")\nax.bar_label(bars, labels=[f\"{x:.1%}\" for x in ranked.field_goal_percentage], padding=3, fontsize=8)\nax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlim(.67, 1.02)\nax.set(title=\"NFL Team Field-Goal Percentage in 2025\", xlabel=\"Field-goal percentage\", ylabel=\"Team\")\ncolorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=.02)\ncolorbar.set_label(\"Winning percentage\"); colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))\nplt.tight_layout(); plt.show()"""),
    md("""The Jets had the highest field-goal percentage (28 of 29, 96.6%) but finished 3-14. Denver finished 14-3 while making 87.5% of its attempts. These examples reinforce that accurate kicking is only one part of team success."""),
    md("""## 4. Conclusion\n\n"
       "The 2025 data show a very weak positive relationship between field-goal percentage and winning percentage, but it is not statistically significant. I cannot conclude that better field-goal percentage caused teams to win more games—or that kicking accuracy has no value. Wins also reflect offense, defense, turnovers, injuries, schedule strength, coaching, and game context."""),
    md("""## 5. Limitations, ethics, and next steps\n\n"
       "This analysis has only 32 observations and uses season aggregates. Raw percentage ignores kick distance, weather, stadium, pressure, blocks, attempt volume, score situation, and kicker changes. One season can also create temporal bias if its patterns are generalized to other eras. The public professional-performance data do not contain private personal information, but the results should not be used to make causal or individual personnel claims. A stronger next study would model attempt-level success using distance, environment, and game situation, then compare actual makes with expected makes."""),
    md("""## 6. References and AI disclosure\n\n"
       "Berry, S. M., & Wood, C. (2004). The cold-foot effect. *Chance, 17*(4), 47-51. https://doi.org/10.1080/09332480.2004.10554926\n\n"
       "Osborne, J. A., & Levine, R. A. (2017). Shrinkage estimation of NFL field goal success probabilities. *Journal of Sports Analytics, 3*(2), 129-146. https://doi.org/10.3233/JSA-16140\n\n"
       "Pasteur, R. D., & Cunningham-Rhoads, K. (2014). An expectation-based metric for NFL field goal kickers. *Journal of Quantitative Analysis in Sports, 10*(1), 49-66. https://doi.org/10.1515/jqas-2013-0039\n\n"
       "Data: nflverse, nflreadpy load functions, https://nflreadpy.nflverse.com/api/load_functions/\n\n"
       "**AI disclosure:** OpenAI ChatGPT with Codex (GPT-5 family), accessed September 12, 2026, helped organize the project, draft and debug code, and revise writing. I reviewed the calculations, documentation, and conclusions and remain responsible for the submitted work. AI was not used as a data source."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUT)
