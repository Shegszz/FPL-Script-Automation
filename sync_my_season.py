
import json
import os
import sys
from datetime import datetime, timezone

import requests
import gspread
from google.oauth2.service_account import Credentials

BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_TEAM_ID = "2551369"
HISTORY_TAB = "My FPL - Season History"
KPI_TAB = "My FPL - KPI Summary"
TIMEOUT = 30

HEADERS = [
    "Season", "Season Status", "Team ID", "Manager", "Team Name",
    "Gameweek", "GW Points", "Transfers Cost (Hits)", "Net GW Points",
    "Cumulative Points", "Overall Rank", "GW Rank", "Team Value",
    "Money In Bank", "Transfers", "Bench Points", "GW Average",
    "GW Highest Score", "Points vs Average", "Last Updated UTC",
]


def get_json(url):
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": "FPL-Season-History-Sync/1.0"},
    )
    response.raise_for_status()
    return response.json()


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
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
                pass

    if not dates:
        raise RuntimeError("Could not determine the current FPL season.")

    start_year = min(dates).year
    if min(dates).month <= 6:
        start_year -= 1

    return f"{start_year}/{str(start_year + 1)[-2:]}"


def connect_spreadsheet():
    raw_credentials = os.getenv("GOOGLE_CREDENTIALS", "").strip()
    spreadsheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()

    if not raw_credentials:
        raise RuntimeError("Missing GOOGLE_CREDENTIALS secret.")
    if not spreadsheet_id:
        raise RuntimeError("Missing GOOGLE_SHEET_ID secret.")

    try:
        info = json.loads(raw_credentials)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_CREDENTIALS is not valid service-account JSON."
        ) from exc

    credentials = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )

    return gspread.authorize(credentials).open_by_key(spreadsheet_id)


def get_worksheet(spreadsheet, title, rows=200, cols=30):
    try:
        return spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(
            title=title, rows=rows, cols=cols
        )


def write_table(worksheet, headers, rows):
    worksheet.clear()
    worksheet.update(
        range_name="A1",
        values=[headers] + rows,
        value_input_option="USER_ENTERED",
    )


def update_kpi_summary(spreadsheet, new_values):
    worksheet = get_worksheet(spreadsheet, KPI_TAB, 20, 40)
    old_data = worksheet.get_all_values()

    # Preserve existing model/KPI columns managed by other scripts.
    old_headers = old_data[0] if old_data else []
    old_row = old_data[1] if len(old_data) > 1 else []

    merged = {}
    for i, header in enumerate(old_headers):
        if header:
            merged[header] = old_row[i] if i < len(old_row) else ""

    merged.update(new_values)

    headers = list(old_headers)
    for key in merged:
        if key not in headers:
            headers.append(key)

    worksheet.clear()
    worksheet.update(
        range_name="A1",
        values=[headers, [merged.get(h, "") for h in headers]],
        value_input_option="USER_ENTERED",
    )


def main():
    team_id = os.getenv("FPL_TEAM_ID", DEFAULT_TEAM_ID).strip()

    if not team_id.isdigit():
        raise RuntimeError(
            f"FPL_TEAM_ID must be numeric; received {team_id!r}."
        )

    print(f"Fetching FPL data for entry {team_id}...")

    bootstrap = get_json(f"{BASE_URL}/bootstrap-static/")
    events = bootstrap.get("events", [])
    if not events:
        raise RuntimeError("FPL API returned no gameweeks.")

    # Confirm the requested ID belongs to a real FPL entry.
    entry = get_json(f"{BASE_URL}/entry/{team_id}/")
    if str(entry.get("id")) != team_id:
        raise RuntimeError("FPL entry ID verification failed.")

    manager = " ".join(
        str(entry.get(key, "")).strip()
        for key in ("player_first_name", "player_last_name")
    ).strip()
    team_name = str(entry.get("name", "")).strip()

    print(f"Verified entry ID: {team_id}")
    print(f"Manager: {manager or 'Unknown'}")
    print(f"Team: {team_name or 'Unknown'}")

    history = get_json(f"{BASE_URL}/entry/{team_id}/history/")
    current = history.get("current", [])

    if not isinstance(current, list):
        raise RuntimeError("Unexpected format in FPL history response.")

    event_by_id = {
        to_int(event.get("id")): event for event in events
    }

    # Keep only gameweeks that exist in the current bootstrap season.
    current = [
        row for row in current
        if to_int(row.get("event")) in event_by_id
    ]
    current.sort(key=lambda row: to_int(row.get("event")))

    if not current:
        raise RuntimeError(
            "The API returned no current-season history. "
            "No sheet data was overwritten. Check the entry ID and API."
        )

    season = get_season(events)
    updated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    rows = []

    for item in current:
        gw = to_int(item.get("event"))
        event = event_by_id[gw]

        points = to_int(item.get("points"))
        hits = to_int(item.get("event_transfers_cost"))
        average = to_int(event.get("average_entry_score"))
        highest = to_int(event.get("highest_score"))

        if event.get("is_current"):
            status = "current"
        elif event.get("is_next"):
            status = "next"
        else:
            status = "played"

        rows.append([
            season,
            status,
            team_id,
            manager,
            team_name,
            gw,
            points,
            hits,
            points - hits,
            to_int(item.get("total_points")),
            to_int(item.get("overall_rank")),
            to_int(item.get("rank")),
            to_int(item.get("value")),
            to_int(item.get("bank")),
            to_int(item.get("event_transfers")),
            to_int(item.get("points_on_bench")),
            average,
            highest,
            points - average,
            updated_at,
        ])

    spreadsheet = connect_spreadsheet()

    history_sheet = get_worksheet(
        spreadsheet, HISTORY_TAB, rows=250, cols=len(HEADERS)
    )
    write_table(history_sheet, HEADERS, rows)

    best = max(rows, key=lambda row: to_int(row[6]))
    latest = rows[-1]

    summary = {
        "Season": season,
        "Season Status": "success",
        "Team ID": team_id,
        "Manager": manager,
        "Team Name": team_name,
        "Gameweeks Recorded": len(rows),
        "Season Points": to_int(latest[9]),
        "Latest GW": to_int(latest[5]),
        "Latest GW Points": to_int(latest[6]),
        "Season Best Score": to_int(best[6]),
        "Season Best GW": to_int(best[5]),
        "Last Updated UTC": updated_at,
        "History Sync Status": "SUCCESS",
        "History Sync Error": "",
    }

    update_kpi_summary(spreadsheet, summary)

    print(f"Season detected: {season}")
    print(f"Gameweeks written: {len(rows)}")
    print(f"Highest recorded GW: {best[5]} ({best[6]} points)")
    print(f"Latest GW: {latest[5]} ({latest[6]} points)")
    print("Season history sync completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SEASON HISTORY SYNC FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
