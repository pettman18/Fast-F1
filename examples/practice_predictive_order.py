"""Predictive Value of Practice Sessions
=====================================

This example compares lap times from FP2 and FP3 sessions against
qualifying and race results. For each race weekend the fastest lap per
Driver in practice is collected. Laps slower than 120% of the average
qualifying lap time are ignored. A Spearman rank correlation is
calculated between the ranking in practice and the final qualifying
and race classification to determine which practice session is most
representative of the weekend result.

This script requires an internet connection on the first run so that
Fast-F1 can download timing data. All downloaded data is cached for
future use.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

import fastf1
from fastf1.core import Laps


fastf1.Cache.enable_cache("./.fastf1cache")


SEASON = 2023


def filter_meaningful(laps: Laps, q_avg: pd.Timedelta) -> Laps:
    """Filter laps slower than 120% of the average qualifying lap time."""
    threshold = q_avg * 1.20
    return laps[laps["LapTime"] <= threshold]


def session_fast_laps(session: fastf1.core.Session, q_avg: pd.Timedelta) -> pd.Series:
    """Return fastest meaningful lap time per driver."""
    laps = filter_meaningful(session.laps, q_avg)
    fastest = laps.pick_fastest("Driver")
    return fastest.set_index("Driver")["LapTime"].sort_values()


def classification_order(session: fastf1.core.Session, column: str) -> pd.Index:
    """Return drivers ordered by a result column."""
    return (
        session.results
        .sort_values(column)["Abbreviation"]
        .reset_index(drop=True)
    )


def evaluate_event(year: int, gp: str) -> pd.DataFrame:
    """Calculate correlation coefficients for one race weekend."""
    fp2 = fastf1.get_session(year, gp, "FP2")
    fp3 = fastf1.get_session(year, gp, "FP3")
    quali = fastf1.get_session(year, gp, "Q")
    race = fastf1.get_session(year, gp, "R")

    fp2.load(); fp3.load(); quali.load(); race.load()

    q_avg = quali.laps["LapTime"].mean()

    practice = {
        "FP2": session_fast_laps(fp2, q_avg),
        "FP3": session_fast_laps(fp3, q_avg),
    }
    quali_order = classification_order(quali, "QFPosition")
    race_order = classification_order(race, "Position")

    data = []
    for name, laps in practice.items():
        common_q = quali_order[quali_order.isin(laps.index)]
        prac_q = laps.index.get_indexer(common_q)
        corr_q, _ = spearmanr(prac_q, range(len(prac_q)))

        common_r = race_order[race_order.isin(laps.index)]
        prac_r = laps.index.get_indexer(common_r)
        corr_r, _ = spearmanr(prac_r, range(len(prac_r)))

        data.append({"session": name, "qualifying": corr_q, "race": corr_r})

    return pd.DataFrame(data)


def analyze_season(year: int = SEASON) -> pd.DataFrame:
    """Evaluate all rounds of a season."""
    schedule = fastf1.get_event_schedule(year)
    result_list = []

    for gp in schedule.EventName:
        try:
            res = evaluate_event(year, gp)
        except Exception as exc:  # pragma: no cover - depends on data availability
            print(f"Skipping {gp}: {exc}")
            continue
        res.insert(0, "Event", gp)
        result_list.append(res)

    return pd.concat(result_list, ignore_index=True)


def main() -> None:
    season_results = analyze_season(SEASON)
    avg = season_results.groupby("session").mean(numeric_only=True)
    print(season_results)
    print("\nAverage correlations:\n", avg)

    best = avg.mean(axis=1).idxmax()
    print(f"\nSession most predictive overall: {best}")


if __name__ == "__main__":
    main()
