"""Build the cleaned team dataset, summary statistics, and portfolio charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"


def build_team_records(games: pd.DataFrame) -> pd.DataFrame:
    regular = games.loc[
        (games["season"] == 2025)
        & (games["game_type"] == "REG")
        & games["home_score"].notna()
        & games["away_score"].notna()
    ].copy()

    home = regular[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"}
    )
    away = regular[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"}
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
    team_games["loss"] = (team_games["points_for"] < team_games["points_against"]).astype(int)
    team_games["tie"] = (team_games["points_for"] == team_games["points_against"]).astype(int)

    records = (
        team_games.groupby("team", as_index=False)
        .agg(
            wins=("win", "sum"),
            losses=("loss", "sum"),
            ties=("tie", "sum"),
            games_played=("team", "size"),
        )
    )
    records["winning_percentage"] = (
        records["wins"] + 0.5 * records["ties"]
    ) / records["games_played"]
    return records


def load_and_clean() -> pd.DataFrame:
    stats = pd.read_csv(RAW / "stats_team_reg_2025.csv")
    games = pd.read_csv(RAW / "games.csv", low_memory=False)
    records = build_team_records(games)

    kicking = stats[
        ["team", "games", "fg_made", "fg_att", "fg_missed", "fg_blocked", "fg_long", "fg_pct"]
    ].copy()
    kicking = kicking.rename(columns={"games": "stats_games", "fg_pct": "source_fg_percentage"})
    kicking["field_goal_percentage"] = kicking["fg_made"] / kicking["fg_att"]

    cleaned = records.merge(kicking, on="team", how="inner", validate="one_to_one")
    cleaned["record"] = (
        cleaned["wins"].astype(str)
        + "-"
        + cleaned["losses"].astype(str)
        + np.where(cleaned["ties"].eq(0), "", "-" + cleaned["ties"].astype(str))
    )
    cleaned = cleaned[
        [
            "team", "wins", "losses", "ties", "record", "games_played",
            "fg_made", "fg_att", "fg_missed", "fg_blocked", "fg_long",
            "field_goal_percentage", "winning_percentage",
        ]
    ].sort_values("field_goal_percentage", ascending=False)

    assert len(cleaned) == 32, "Expected one row for each NFL team."
    assert cleaned["team"].is_unique, "Team abbreviations must be unique."
    assert cleaned.isna().sum().sum() == 0, "Clean analysis columns should have no missing values."
    assert (cleaned["games_played"] == 17).all(), "Each team should have 17 regular-season games."
    assert cleaned["wins"].sum() + cleaned["ties"].sum() / 2 == 272, "League record totals are inconsistent."
    return cleaned


def save_charts(df: pd.DataFrame, regression) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    navy, teal, gold = "#14213D", "#007C91", "#FCA311"

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.regplot(
        data=df,
        x="field_goal_percentage",
        y="winning_percentage",
        ci=None,
        scatter_kws={"s": 75, "alpha": 0.82, "color": teal, "edgecolor": "white"},
        line_kws={"color": gold, "linewidth": 2.5},
        ax=ax,
    )
    for row in df.itertuples():
        predicted = regression.intercept + regression.slope * row.field_goal_percentage
        if (
            row.field_goal_percentage == df["field_goal_percentage"].max()
            or row.field_goal_percentage == df["field_goal_percentage"].min()
            or abs(row.winning_percentage - predicted) >= 0.19
        ):
            ax.annotate(row.team, (row.field_goal_percentage, row.winning_percentage),
                        xytext=(5, 5), textcoords="offset points", fontsize=9, color=navy)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(
        title="Field-Goal Accuracy and Team Winning Percentage, 2025 NFL Regular Season",
        xlabel="Team field-goal percentage",
        ylabel="Team winning percentage",
    )
    ax.text(
        0.01, 0.99,
        f"Pearson r = {regression.rvalue:.2f}  |  R² = {regression.rvalue ** 2:.2f}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    sns.despine()
    fig.tight_layout()
    fig.savefig(ASSETS / "fg_pct_vs_win_pct.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ranked = df.sort_values("field_goal_percentage")
    fig, ax = plt.subplots(figsize=(11, 12))
    norm = plt.Normalize(df["winning_percentage"].min(), df["winning_percentage"].max())
    cmap = plt.colormaps["viridis"]
    bars = ax.barh(
        ranked["team"], ranked["field_goal_percentage"],
        color=cmap(norm(ranked["winning_percentage"])), edgecolor="white",
    )
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in ranked["field_goal_percentage"]],
                 padding=3, fontsize=8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(max(0.60, ranked["field_goal_percentage"].min() - 0.04), 1.02)
    ax.set(
        title="NFL Team Field-Goal Percentage in 2025",
        xlabel="Field-goal percentage",
        ylabel="Team",
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Winning percentage")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    sns.despine()
    fig.tight_layout()
    fig.savefig(ASSETS / "team_fg_pct_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df.to_csv(PROCESSED / "nfl_2025_field_goal_team_analysis.csv", index=False)

    pearson_r, pearson_p = pearsonr(df["field_goal_percentage"], df["winning_percentage"])
    spearman_rho, spearman_p = spearmanr(df["field_goal_percentage"], df["winning_percentage"])
    regression = linregress(df["field_goal_percentage"], df["winning_percentage"])
    save_charts(df, regression)

    summary = {
        "teams": int(len(df)),
        "regular_season_games": 272,
        "missing_analysis_values": int(df.isna().sum().sum()),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "r_squared": regression.rvalue ** 2,
        "slope": regression.slope,
        "intercept": regression.intercept,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "highest_fg_team": df.loc[df["field_goal_percentage"].idxmax(), "team"],
        "highest_fg_pct": df["field_goal_percentage"].max(),
        "lowest_fg_team": df.loc[df["field_goal_percentage"].idxmin(), "team"],
        "lowest_fg_pct": df["field_goal_percentage"].min(),
        "highest_win_team": df.loc[df["winning_percentage"].idxmax(), "team"],
        "highest_win_pct": df["winning_percentage"].max(),
    }
    (PROCESSED / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(df.to_string(index=False, formatters={
        "field_goal_percentage": "{:.3f}".format,
        "winning_percentage": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()
