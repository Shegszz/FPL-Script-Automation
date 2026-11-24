import json
import os
import sys
from datetime import datetime, timezone

import requests
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

BASE_URL = "https://fantasy.premierleague.com/api"
TEAM_ID = os.getenv("FPL_TEAM_ID", "2551369").strip()
EXPECTED_SEASON = "2026/27"
HISTORY_TAB = "My FPL - Season History"
KPI_TAB = "My FPL - KPI Summary"
TIMEOUT = 30

HISTORY_COLUMNS = [
    "Season", "Season Status", "Team ID", "Gameweek", "GW Points",
    "Transfers Cost (Hits)", "Net GW Points", "Cumulative Points",
    "Overall Rank", "GW Rank", "Team Value", "Money In Bank",
    "Total Transfers Made", "Hit Taken", "Points On Bench", "GW Average",
    "GW Highest Score", "Points vs Average", "Points vs TOTW",
    "Beat Average", "Chip Used", "Rank Change", "Rank Direction",
    "Rank Delta Display", "Points Delta Display", "vs Average Display",
    "Last Updated",
]


def api_get(endpoint):
    response = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        timeout=TIMEOUT,
        headers={"User-Agent": "FPL-Analytics-Engine/2.0"},
    )
    response.raise_for_status()
    return response.json()


def number(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def season_from_events(events):
    deadlines = []
    for event in events:
        value = event.get("deadline_time")
        if value:
            try:
                deadlines.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
            except (TypeError, ValueError):
                continue
    if not deadlines:
        raise RuntimeError("FPL bootstrap did not contain usable event deadlines.")
    first = min(deadlines)
    start_year = first.year if first.month >= 7 else first.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def connect_spreadsheet():
    raw_credentials = os.getenv("GOOGLE_CREDENTIALS", "").strip()
    spreadsheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if not raw_credentials:
        raise RuntimeError("Missing GOOGLE_CREDENTIALS environment variable.")
    if not spreadsheet_id:
        raise RuntimeError("Missing GOOGLE_SHEET_ID environment variable.")

    try:
        info = json.loads(raw_credentials)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_CREDENTIALS is not valid JSON.") from exc

    credentials = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(credentials).open_by_key(spreadsheet_id)


def get_or_create_sheet(book, title, rows=300, cols=40):
    try:
        return book.worksheet(title)
    except gspread.WorksheetNotFound:
        return book.add_worksheet(title=title, rows=rows, cols=cols)


def write_history(book, dataframe):
    worksheet = get_or_create_sheet(
        book, HISTORY_TAB, rows=300, cols=len(HISTORY_COLUMNS)
    )
    worksheet.clear()
    set_with_dataframe(
        worksheet,
        dataframe[HISTORY_COLUMNS],
        include_index=False,
        include_column_header=True,
        resize=True,
    )
    worksheet.freeze(rows=1, cols=3)


def update_kpi(book, updates):
    worksheet = get_or_create_sheet(book, KPI_TAB, rows=20, cols=50)
    old = worksheet.get_all_values()
    old_headers = old[0] if old else []
    old_row = old[1] if len(old) > 1 else []

    # Keep model metrics produced by fpl_update.py; update only season facts.
    merged = {
        header: old_row[i] if i < len(old_row) else ""
        for i, header in enumerate(old_headers)
        if header
    }
    merged.update(updates)

    headers = list(old_headers)
    for key in merged:
        if key not in headers:
            headers.append(key)

    worksheet.clear()
    worksheet.update(
        range_name="A1",
        values=[headers, [merged.get(header, "") for header in headers]],
        value_input_option="USER_ENTERED",
    )
    worksheet.freeze(rows=1)


def main():
    if not TEAM_ID.isdigit():
        raise RuntimeError(f"FPL_TEAM_ID must be numeric; received {TEAM_ID!r}.")

    bootstrap = api_get("bootstrap-static/")
    events = bootstrap.get("events", [])
    if not events:
        raise RuntimeError("FPL bootstrap returned no events.")

    season = season_from_events(events)
    if season != EXPECTED_SEASON:
        raise RuntimeError(
            f"Expected season {EXPECTED_SEASON}, but FPL API reports {season}. "
            "No Google Sheets data was changed."
        )

    entry = api_get(f"entry/{TEAM_ID}/")
    if str(entry.get("id")) != TEAM_ID:
        raise RuntimeError("FPL manager ID verification failed.")

    manager = " ".join(
        str(entry.get(key, "")).strip()
        for key in ("player_first_name", "player_last_name")
    ).strip()
    team_name = str(entry.get("name", "")).strip()

    print(f"Verified FPL entry: {TEAM_ID} | {manager} | {team_name}")

    history = api_get(f"entry/{TEAM_ID}/history/")
    current = history.get("current", [])
    if not isinstance(current, list):
        raise RuntimeError("FPL history endpoint returned an invalid current list.")

    event_by_id = {number(event.get("id")): event for event in events}
    current = [
        row for row in current
        if number(row.get("event")) in event_by_id
    ]
    current.sort(key=lambda row: number(row.get("event")))

    chips = {
        number(chip.get("event")): str(chip.get("name", "")).replace("_", " ").title()
        for chip in history.get("chips", [])
    }

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    previous_rank = None

    for item in current:
        gw = number(item.get("event"))
        event = event_by_id[gw]
        points = number(item.get("points"))
        hits = number(item.get("event_transfers_cost"))
        rank = number(item.get("overall_rank"))
        average = number(event.get("average_entry_score"))
        highest = number(event.get("highest_score"))

        rank_change = (
            previous_rank - rank
            if previous_rank is not None and previous_rank > 0 and rank > 0
            else 0
        )
        if rank > 0:
            previous_rank = rank

        vs_average = points - average
        vs_totw = points - highest
        rank_direction = (
            "↑ Improved" if rank_change > 0
            else "↓ Dropped" if rank_change < 0
            else "→ Held"
        )
        rank_display = (
            f"↑ +{rank_change:,}" if rank_change > 0
            else f"↓ {rank_change:,}" if rank_change < 0
            else "→ No change"
        )

        rows.append({
            "Season": season,
            "Season Status": "in_progress",
            "Team ID": TEAM_ID,
            "Gameweek": gw,
            "GW Points": points,
            "Transfers Cost (Hits)": hits,
            "Net GW Points": points - hits,
            # FPL's total_points is authoritative and includes transfer hits.
            "Cumulative Points": number(item.get("total_points")),
            "Overall Rank": rank,
            "GW Rank": number(item.get("rank")),
            "Team Value": round(number(item.get("value")) / 10, 1),
            "Money In Bank": round(number(item.get("bank")) / 10, 1),
            "Total Transfers Made": number(item.get("event_transfers")),
            "Hit Taken": "Yes ❌" if hits > 0 else "No ✅",
            "Points On Bench": number(item.get("points_on_bench")),
            "GW Average": average,
            "GW Highest Score": highest,
            "Points vs Average": vs_average,
            "Points vs TOTW": vs_totw,
            "Beat Average": "Yes ✅" if vs_average > 0 else "No ❌",
            "Chip Used": chips.get(gw, ""),
            "Rank Change": rank_change,
            "Rank Direction": rank_direction,
            "Rank Delta Display": rank_display,
            "Points Delta Display": f"+{points} pts this GW",
            "vs Average Display": (
                f"+{vs_average} vs avg" if vs_average > 0
                else f"{vs_average} vs avg"
            ),
            "Last Updated": updated_at,
        })

    # Determine the live/upcoming GW from bootstrap, not the last completed GW.
    active_event = next(
        (event for event in events if event.get("is_current")),
        next((event for event in events if event.get("is_next")), None),
    )
    if active_event is None:
        active_event = next(
            (event for event in reversed(events) if event.get("finished")),
            events[0],
        )
    current_gameweek = number(active_event.get("id"))

    if rows:
        history_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
        latest = rows[-1]
        best = max(rows, key=lambda row: row["GW Points"])
        best_rank = min(
            (row["Overall Rank"] for row in rows if row["Overall Rank"] > 0),
            default=0,
        )
        season_status = "in_progress"
        total_points = number(latest["Cumulative Points"])
        overall_rank = number(latest["Overall Rank"])
        latest_gw_points = number(latest["GW Points"])
        latest_net_points = number(latest["Net GW Points"])
        best_score = number(best["GW Points"])
        best_gw = number(best["Gameweek"])
        total_hits = sum(number(row["Transfers Cost (Hits)"]) for row in rows)
        bench_total = sum(number(row["Points On Bench"]) for row in rows)
        beat_average = sum(1 for row in rows if row["Points vs Average"] > 0)
    else:
        # Valid current-season API response, but no GW has been completed.
        # Write an explicit placeholder so 2025/26 cannot remain on display.
        season_status = "not_started"
        history_df = pd.DataFrame([{
            "Season": season,
            "Season Status": "not_started",
            "Team ID": TEAM_ID,
            "Gameweek": current_gameweek,
            "GW Points": "",
            "Last Updated": updated_at,
        }])
        latest_gw_points = 0
        latest_net_points = 0
        total_points = 0
        overall_rank = 0
        best_score = 0
        best_gw = 0
        best_rank = 0
        total_hits = 0
        bench_total = 0
        beat_average = 0

    # Verify all real rows are explicitly from the expected season.
    if "Season" not in history_df.columns:
        raise RuntimeError("Internal validation: Season column missing.")
    if not (history_df["Season"].astype(str) == EXPECTED_SEASON).all():
        raise RuntimeError("Internal validation failed: mixed/incorrect seasons.")

    book = connect_spreadsheet()
    write_history(book, history_df)

    update_kpi(book, {
        "Season": season,
        "Season Status": season_status,
        "Team ID": TEAM_ID,
        "Manager": manager,
        "Team Name": team_name,
        "Total Points": total_points,
        "Overall Rank": overall_rank,
        "Current Gameweek": current_gameweek,
        "GW Points": latest_gw_points,
        "Net GW Points": latest_net_points,
        "Season Best Score": best_score,
        "Season Best GW": best_gw,
        "Season Best Rank": best_rank,
        "Season Total Hits Cost": total_hits,
        "Season Points On Bench": bench_total,
        "GWs Beat Average": beat_average,
        "Gameweeks Recorded": len(rows),
        "Last Updated": updated_at,
        "History Sync Status": "SUCCESS",
        "History Sync Error": "",
    })

    print(f"Season: {season}")
    print(f"Team ID: {TEAM_ID}")
    print(f"Manager: {manager} | Team: {team_name}")
    print(f"Current/next GW: {current_gameweek}")
    print(f"History rows written: {len(rows)}")
    if rows:
        print(f"Best GW: GW{best_gw} - {best_score} points")
        print(f"Latest completed GW: {rows[-1]['Gameweek']} - {latest_gw_points} points")
    print("SUCCESS: My Season tabs refreshed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MY SEASON SYNC FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
