"""Qualifying results overview
=============================

Compare fastest laps from practice and qualifying sessions across the
2024 season and evaluate which session best predicts the final race
result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from timple.timedelta import strftimedelta

import fastf1
import fastf1.plotting
from fastf1.core import Laps

fastf1.Cache.enable_cache("./.fastf1cache")

# Enable Matplotlib patches for plotting timedelta values
fastf1.plotting.setup_mpl(
    mpl_timedelta_support=True,
    misc_mpl_mods=False,
    color_scheme=None,
)

SEASON = 2024
SESSIONS_TO_COMPARE = ["FP1", "FP2", "FP3", "Q"]


def fastest_lap_info(session: fastf1.core.Session) -> tuple[Laps, pd.Series, list[str]]:
    """Return fastest laps and pole information for plotting."""
    drivers = pd.unique(session.laps["Driver"])
    list_fastest_laps = [
        session.laps.pick_drivers(drv).pick_fastest() for drv in drivers
    ]
    fastest_laps = (
        Laps(list_fastest_laps).sort_values(by="LapTime").reset_index(drop=True)
    )
    pole_lap = fastest_laps.pick_fastest()
    fastest_laps["LapTimeDelta"] = fastest_laps["LapTime"] - pole_lap["LapTime"]
    team_colors = [
        fastf1.plotting.get_team_color(lap["Team"], session=session)
        for _, lap in fastest_laps.iterlaps()
    ]
    return fastest_laps, pole_lap, team_colors


def plot_session(session: fastf1.core.Session) -> None:
    """Plot fastest lap times of one session."""
    fastest_laps, pole_lap, team_colors = fastest_lap_info(session)

    fig, ax = plt.subplots()
    ax.barh(
        fastest_laps.index,
        fastest_laps["LapTimeDelta"],
        color=team_colors,
        edgecolor="grey",
    )
    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps["Driver"])

    ax.invert_yaxis()  # show fastest at the top
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", linestyle="--", color="black", zorder=-1000)

    lap_time_string = strftimedelta(pole_lap["LapTime"], "%m:%s.%ms")
    plt.suptitle(
        f"{session.event['EventName']} {session.event.year} {session.name}\n"
        f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})"
    )
    plt.show()


def session_result_order(session: fastf1.core.Session) -> pd.Index:
    """Return driver order by fastest lap."""
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    fastest = session.laps.pick_fastest("Driver")
    return fastest.set_index("Driver")["LapTime"].sort_values().index


def race_result_order(session: fastf1.core.Session) -> pd.Index:
    """Return finishing order of a race."""
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session.results.sort_values("Position")["Abbreviation"].reset_index(
        drop=True
    )


def evaluate_event(event: pd.Series) -> pd.DataFrame:
    """Calculate correlation coefficients for one race weekend."""
    race = fastf1.get_session(SEASON, event["EventName"], "R")
    race_order = race_result_order(race)

    data = []
    for name in SESSIONS_TO_COMPARE:
        try:
            sess = fastf1.get_session(SEASON, event["EventName"], name)
            order = session_result_order(sess)
        except Exception as exc:  # pragma: no cover - data may be missing
            print(f"Skipping {name} for {event['EventName']}: {exc}")
            continue

        common = race_order[race_order.isin(order)]
        prac_idx = order.get_indexer(common)
        race_idx = range(len(common))
        corr, _ = spearmanr(prac_idx, race_idx)
        data.append({"session": name, "correlation": corr})

    return pd.DataFrame(data)


def main() -> None:
    schedule = fastf1.get_event_schedule(SEASON)

    # Plot an example for the first event
    first_event = schedule.iloc[0]["EventName"]
    for sess_name in SESSIONS_TO_COMPARE:
        sess = fastf1.get_session(SEASON, first_event, sess_name)
        sess.load()
        plot_session(sess)

    # Evaluate predictive value for the full season
    result_list = []
    for rnd in range(1, 25):
        ev = schedule[schedule["RoundNumber"] == rnd]
        if ev.empty:
            print(f"Round {rnd} not found in schedule")
            continue
        ev = ev.iloc[0]
        print(f"Processing {ev['EventName']}")
        res = evaluate_event(ev)
        res.insert(0, "Event", ev["EventName"])
        result_list.append(res)

    season_results = pd.concat(result_list, ignore_index=True)
    avg = season_results.groupby("session").mean(numeric_only=True)
    print(season_results)
    print("\nAverage correlations:\n", avg)

    best = avg["correlation"].idxmax()
    print(f"\nSession most predictive overall: {best}")


if __name__ == "__main__":
    main()
