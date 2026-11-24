
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
HISTORY_TAB = "My FPL - Season History"
KPI_TAB = "My FPL - KPI Summary"
EXPECTED_SEASON = "2026/27"


def api_get(endpoint):
    response = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        timeout=30,
        headers={"User-Agent": "FPL-Analytics-Engine/1.0"},
    )
    response.raise_for_status()
    return response.json()


def number(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_season(events):
    dates = []
    for event in events:
        deadline = event.get("deadline_time")
        if deadline:
            try:
                dates.append(
                    datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                )
            except ValueError:
                continue

    if not dates:
        raise RuntimeError("Could not determine season from FPL events.")

    first = min(dates)
    year = first.year if first.month >= 7 else first.year - 1
    return f"{year}/{str(year + 1)[-2:]}"


def connect_sheet():
    raw = os.getenv("GOOGLE_CREDENTIALS", "")
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "")

    if not raw or not sheet_id:
        raise RuntimeError("Missing Google Sheets credentials or sheet ID.")

    info = json.loads(raw)
    credentials = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(credentials).open_by_key(sheet_id)


def get_or_create_sheet(book, title, rows=300, cols=40):
    try:
        return book.worksheet(title)
    except gspread.WorksheetNotFound:
        return book.add_worksheet(title=title, rows=rows, cols=cols)


def write_dataframe(book, title, dataframe):
    ws = get_or_create_sheet(book, title)
    ws.clear()
    set_with_dataframe(
        ws,
        dataframe,
        include_index=False,
        include_column_header=True,
        resize=True,
    )
    ws.freeze(rows=1)


def update_kpi(book, values):
    ws = get_or_create_sheet(book, KPI_TAB)
    existing = ws.get_all_values()

    old_headers = existing[0] if existing else []
    old_row = existing[1] if len(existing) > 1 else []

    merged = {}
    for i, header in enumerate(old_headers):
        if header:
            merged[header] = old_row[i] if i < len(old_row) else ""

    merged.update(values)

    headers = list(old_headers)
    for key in merged:
        if key not in headers:
            headers.append(key)

    ws.clear()
    ws.update(
        range_name="A1",
        values=[headers, [merged.get(h, "") for h in headers]],
        value_input_option="USER_ENTERED",
    )


def main():
    if not TEAM_ID.isdigit():
        raise RuntimeError(f"Invalid FPL_TEAM_ID: {TEAM_ID}")

    print(f"Checking FPL entry {TEAM_ID}")

    bootstrap = api_get("bootstrap-static/")
    events = bootstrap.get("events", [])

    if not events:
        raise RuntimeError("No FPL events returned by the API.")

    season = get_season(events)

    if season != EXPECTED_SEASON:
        raise RuntimeError(
            f"Expected {EXPECTED_SEASON}, but FPL API indicates {season}. "
            "Stopping to avoid writing the wrong season."
        )

    entry = api_get(f"entry/{TEAM_ID}/")

    if str(entry.get("id")) != TEAM_ID:
        raise RuntimeError("FPL entry ID verification failed.")

    manager = " ".join([
        str(entry.get("player_first_name", "")).strip(),
        str(entry.get("player_last_name", "")).strip(),
    ]).strip()

    team_name = entry.get("name", "")

    history = api_get(f"entry/{TEAM_ID}/history/")
    current = history.get("current", [])

    if not isinstance(current, list) or not current:
        raise RuntimeError(
            "No current-season gameweek history returned. "
            "Existing Google Sheets data has not been overwritten."
        )

    event_lookup = {
        number(event.get("id")): event for event in events
    }

    current = [
        row for row in current
        if number(row.get("event")) in event_lookup
    ]
    current.sort(key=lambda row: number(row.get("event")))

    if not current:
        raise RuntimeError("No valid current-season gameweeks found.")

    chips = {
        number(chip.get("event")): chip.get("name", "").replace("_", " ").title()
        for chip in history.get("chips", [])
    }

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []

    previous_rank = None

    for item in current:
        gw = number(item.get("event"))
        event = event_lookup[gw]

        points = number(item.get("points"))
        hits = number(item.get("event_transfers_cost"))
        rank = number(item.get("overall_rank"))

        average = number(event.get("average_entry_score"))
        highest = number(event.get("highest_score"))

        rank_change = (
            previous_rank - rank
            if previous_rank is not None and rank
            else 0
        )
        previous_rank = rank if rank else previous_rank

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

        vs_average = points - average
        vs_highest = points - highest

        if event.get("is_current"):
            status = "current"
        elif event.get("is_next"):
            status = "next"
        else:
            status = "played"

        rows.append({
            "Season": season,
            "Season Status": "in_progress",
            "Team ID": TEAM_ID,
            "Manager": manager,
            "Team Name": team_name,
            "Gameweek": gw,
            "GW Points": points,
            "Transfers Cost (Hits)": hits,
            "Net GW Points": points - hits,
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
            "Points vs TOTW": vs_highest,
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
            "Last Updated": now,
        })

    dataframe = pd.DataFrame(rows)

    # Safety check: never publish a history table with another season's rows.
    if not (dataframe["Season"] == EXPECTED_SEASON).all():
        raise RuntimeError("Season validation failed. Sheet not updated.")

    book = connect_sheet()

    # Replace the entire season history with verified current-season records.
    write_dataframe(book, HISTORY_TAB, dataframe)

    latest = rows[-1]
    best = max(rows, key=lambda row: row["GW Points"])

    # Preserve the other KPI fields produced by fpl_update.py.
    update_kpi(book, {
        "Season": season,
        "Season Status": "in_progress",
        "Team ID": TEAM_ID,
        "Manager": manager,
        "Team Name": team_name,
        "Total Points": number(latest["Cumulative Points"]),
        "Overall Rank": number(latest["Overall Rank"]),
        "Current Gameweek": number(latest["Gameweek"]),
        "Season Best Score": number(best["GW Points"]),
        "Season Best GW": number(best["Gameweek"]),
        "Gameweeks Recorded": len(rows),
        "Last Updated": now,
        "History Sync Status": "SUCCESS",
        "History Sync Error": "",
    })

    print("------------------------------------")
    print(f"SEASON: {season}")
    print(f"TEAM ID: {TEAM_ID}")
    print(f"MANAGER: {manager}")
    print(f"GAMEWEEKS: {len(rows)}")
    print(f"BEST GW: GW{best['Gameweek']} - {best['GW Points']} points")
    print(f"LATEST GW: GW{latest['Gameweek']}")
    print("SUCCESS: Current season history updated.")
    print("------------------------------------")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SEASON SYNC FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
