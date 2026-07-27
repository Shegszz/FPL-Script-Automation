# =============================================================================
# fpl_update.py
# FPL Analytics Engine — Daily Data Pipeline
# Author: Segun Bakare | github.com/Shegszz/FPL-Script-Automation
#
# Pipeline:
#   FPL API → Python ETL → Google Sheets (auto-refreshed daily via GitHub Actions)
#
# Sheets written (General — all viewers):
#   - Player Data
#   - Smart Picks - Goalkeepers
#   - Smart Picks - Defenders
#   - Smart Picks - Midfielders
#   - Smart Picks - Forwards
#   - Best Attacking Teams
#   - Best Defensive Teams
#   - Fixture Planner          ← NEW: FDR heatmap, DGW/BGW, avg FDR next 8 GWs
#   - FPL Key Metrics Guide
#
# Sheets written (Personal — My Season page):
#   - My FPL - Season History  ← UPDATED: KPI deltas, rank arrows, xP vs actual
#   - My FPL - Current Squad   ← UPDATED: free transfer logic, xP vs actual
#   - My FPL - Transfers
#   - My FPL - KPI Summary     ← NEW: single-row KPI card feed for dashboard
# =============================================================================

import os
import json
import re
import time

import gspread
import numpy as np
import pandas as pd
import requests

from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials

# ⭐ PRODUCTION MODEL IMPORT — function name matches exactly
from fpl_ml_model import add_ml_predictions_v2

# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------

creds_json = os.getenv("GOOGLE_CREDENTIALS")
if not creds_json:
    raise EnvironmentError("GOOGLE_CREDENTIALS environment variable not set.")

creds_dict = json.loads(creds_json)

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(credentials)

sheet_id = os.getenv("GOOGLE_SHEET_ID")
if not sheet_id:
    raise EnvironmentError("GOOGLE_SHEET_ID environment variable not set.")

sheet = client.open_by_key(sheet_id)

# ---------------------------------------------------------------------------
# MY FPL TEAM ID
# Reads from GitHub Actions secret first.
# Falls back to default (your public FPL Team ID) for local runs.
# FPL Team IDs are publicly visible on the FPL website — not sensitive.
# ---------------------------------------------------------------------------

MY_TEAM_ID = os.getenv("FPL_TEAM_ID", "2551369")

# ---------------------------------------------------------------------------
# FDR COLOUR MAPPING
# Exact FPL official colour codes (hex) per difficulty rating.
# DGW = Double Gameweek (team plays twice — gold opportunity).
# BGW = Blank Gameweek (team doesn't play — avoid before it hits).
# ---------------------------------------------------------------------------

FDR_COLOURS = {
    1: '#375523',   # FPL dark green  — very easy
    2: '#01FC7A',   # FPL bright green — easy
    3: '#E7E7E7',   # FPL light grey  — medium
    4: '#FF1751',   # FPL red         — hard
    5: '#80072D',   # FPL dark red    — very hard
    'DGW': '#DAA520',  # Gold  — double gameweek (custom, not FPL)
    'BGW': '#4A4A4A',  # Dark grey — blank gameweek (custom)
}

FDR_LABELS = {
    1: '1 — Very Easy',
    2: '2 — Easy',
    3: '3 — Medium',
    4: '4 — Hard',
    5: '5 — Very Hard',
    'DGW': 'DGW ⚡',
    'BGW': 'BGW —',
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def write_to_sheet(sheet, df: pd.DataFrame, sheet_name: str) -> None:
    """Write a DataFrame to a named worksheet, creating it if it doesn't exist."""
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="2000", cols="60")

    # A DataFrame built from an empty list of rows (e.g. pd.DataFrame([]))
    # has 0 columns as well as 0 rows. gspread_dataframe resizes the sheet
    # to the DataFrame's exact shape, and asking Google Sheets for 0
    # columns is an invalid request. A single "no data yet" placeholder
    # column keeps this safe while still correctly clearing out stale rows
    # from a previous run — e.g. a personal sheet during preseason.
    if df.shape[1] == 0:
        df = pd.DataFrame({'Status': ['No data yet']})

    worksheet.clear()
    set_with_dataframe(worksheet, df, include_column_header=True, resize=True)
    worksheet.freeze(rows=1, cols=min(2, len(df.columns)))
    print(f"   ✅ Written: '{sheet_name}' ({len(df)} rows × {len(df.columns)} cols)")


def fetch_fpl_data(url: str, max_retries: int = 4, backoff: float = 2.0):
    """Fetch FPL API endpoint with exponential back-off retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = backoff ** (attempt + 2)
                print(f"   ⏳ Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ HTTP error: {e}")
                return None
        except (requests.exceptions.RequestException, ValueError) as e:
            wait = backoff ** attempt
            print(f"   ⚠️  Attempt {attempt+1}/{max_retries} failed: {e}. Retry in {wait:.0f}s...")
            time.sleep(wait)
    print(f"   ❌ Failed to fetch {url} after {max_retries} attempts.")
    return None


def parse_difficulty(val) -> float:
    """Safely parse a fixture difficulty value regardless of type."""
    if isinstance(val, list):
        valid = [x for x in val if x not in (None, np.nan, '')]
        return float(sum(valid)) if valid else 0.0
    if isinstance(val, str):
        try:
            return sum(float(x.strip()) for x in val.split(',') if x.strip())
        except ValueError:
            return 0.0
    if isinstance(val, (int, float)) and not np.isnan(float(val)):
        return float(val)
    return 0.0


def gw_col_number(colname: str) -> int:
    """Extract integer from 'Gameweek N' column name for sorting."""
    m = re.search(r'Gameweek\s+(\d+)', colname)
    return int(m.group(1)) if m else 0


def fdr_label(fdr_val: int, is_dgw: bool = False, is_bgw: bool = False) -> str:
    """Return display label for a fixture cell including DGW/BGW flags."""
    if is_bgw:
        return 'BGW —'
    if is_dgw:
        return 'DGW ⚡'
    return str(fdr_val) if fdr_val else '—'


def fdr_hex(fdr_val, is_dgw: bool = False, is_bgw: bool = False) -> str:
    """Return the FPL-official hex colour for a difficulty rating."""
    if is_bgw:
        return FDR_COLOURS['BGW']
    if is_dgw:
        return FDR_COLOURS['DGW']
    return FDR_COLOURS.get(int(fdr_val), '#E7E7E7')

# ---------------------------------------------------------------------------
# 1. BOOTSTRAP DATA
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("🚀 FPL DATA PIPELINE STARTING")
print("="*70)

fpl_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
print("\n📡 Fetching bootstrap data...")
data = fetch_fpl_data(fpl_url)
if data is None:
    raise RuntimeError("Failed to fetch bootstrap-static. Pipeline aborted.")

players_raw = data['elements']
teams       = {team['id']: team['short_name'] for team in data['teams']}
positions   = {pos['id']: pos['singular_name'] for pos in data['element_types']}
events      = data['events']
teams_data  = data['teams']   # Full team list — passed to ML model

# ---------------------------------------------------------------------------
# 2. CURRENT GAMEWEEK + NEXT 8
# Extended from next 5 to next 8 to support the FDR heatmap (8 GW lookahead).
# The player data columns still use next 5 only (bandwidth / sheet size).
# ---------------------------------------------------------------------------

current_gameweek = next(
    (e['id'] for e in events if e['is_current']),
    next((e['id'] for e in events if e['is_next']), None)
)

if current_gameweek is None:
    current_gameweek = max(e['id'] for e in events if e.get('finished'))

print(f"   ✅ Current Gameweek: {current_gameweek}")

# ---------------------------------------------------------------------------
# SEASON LABEL — derived from GW1's real deadline date, never the system clock
# or a hardcoded string. The PL season always starts in August, so GW1's
# deadline year is the authoritative "season start year" straight from the
# FPL API. This means the label auto-advances every season with zero code
# changes — no more "2025/26" baked in somewhere and forgotten.
# ---------------------------------------------------------------------------
gw1_deadline = next((e.get('deadline_time') for e in events if e['id'] == 1), None)
season_start_year = int(gw1_deadline[:4]) if gw1_deadline else pd.Timestamp.now().year
season_label = f"{season_start_year}/{str(season_start_year + 1)[-2:]}"
print(f"   ✅ Season: {season_label}")

# Next 5 GWs — for player columns in Player Data sheet
next_5_gameweeks = [
    {'id': e['id'], 'name': e['name']}
    for e in events
    if e['id'] > current_gameweek
][:5]

# Next 8 GWs — for FDR heatmap in Fixture Planner sheet
next_8_gameweeks = [
    {'id': e['id'], 'name': e['name']}
    for e in events
    if e['id'] > current_gameweek
][:8]

# GW average points lookup — used for xP vs Average comparisons
gw_averages = {e['id']: e.get('average_entry_score', 0) for e in events}

# GW highest scores — used for Team of the Week comparison
gw_highest  = {e['id']: e.get('highest_score', 0) for e in events}

print(f"   ✅ Next 5 GWs: {[g['name'] for g in next_5_gameweeks]}")
print(f"   ✅ Next 8 GWs: {[g['name'] for g in next_8_gameweeks]}")

# ---------------------------------------------------------------------------
# 3. FIXTURES
# Builds TWO lookups:
#   team_opponents     → next 5 GWs (for Player Data columns)
#   team_opponents_8gw → next 8 GWs (for Fixture Planner heatmap)
# Also detects Double Gameweeks (team appears twice in one GW)
# and Blank Gameweeks (team has no fixture in a GW).
# ---------------------------------------------------------------------------

print("\n📡 Fetching fixtures...")
fixtures = fetch_fpl_data('https://fantasy.premierleague.com/api/fixtures/')
if fixtures is None:
    print("   ⚠️  No fixtures data. Continuing with empty fixtures.")
    fixtures = []

fixtures_data = fixtures   # Passed to ML model

next_5_ids = {gw['id'] for gw in next_5_gameweeks}
next_8_ids = {gw['id'] for gw in next_8_gameweeks}

# 5-GW lookup (player data columns)
team_opponents = {
    team_id: {
        gw['id']: {'opponent': [], 'difficulty': [], 'home_away': []}
        for gw in next_5_gameweeks
    }
    for team_id in teams.keys()
}

# 8-GW lookup (fixture planner heatmap)
team_opponents_8gw = {
    team_id: {
        gw['id']: {'opponent': [], 'difficulty': [], 'home_away': [], 'fixture_count': 0}
        for gw in next_8_gameweeks
    }
    for team_id in teams.keys()
}

for fixture in fixtures:
    event_id     = fixture.get('event')
    home_team_id = fixture.get('team_h')
    away_team_id = fixture.get('team_a')
    home_diff    = fixture.get('team_h_difficulty')
    away_diff    = fixture.get('team_a_difficulty')

    if not (event_id and home_team_id and away_team_id):
        continue

    # 5-GW lookup
    if event_id in next_5_ids:
        if home_team_id in team_opponents and event_id in team_opponents[home_team_id]:
            team_opponents[home_team_id][event_id]['opponent'].append(teams[away_team_id])
            team_opponents[home_team_id][event_id]['difficulty'].append(home_diff)
            team_opponents[home_team_id][event_id]['home_away'].append(f"{teams[home_team_id]}(H)")

        if away_team_id in team_opponents and event_id in team_opponents[away_team_id]:
            team_opponents[away_team_id][event_id]['opponent'].append(teams[home_team_id])
            team_opponents[away_team_id][event_id]['difficulty'].append(away_diff)
            team_opponents[away_team_id][event_id]['home_away'].append(f"{teams[away_team_id]}(A)")

    # 8-GW lookup
    if event_id in next_8_ids:
        if home_team_id in team_opponents_8gw and event_id in team_opponents_8gw[home_team_id]:
            d = team_opponents_8gw[home_team_id][event_id]
            d['opponent'].append(teams[away_team_id])
            d['difficulty'].append(home_diff)
            d['home_away'].append(f"{teams[home_team_id]}(H)")
            d['fixture_count'] += 1

        if away_team_id in team_opponents_8gw and event_id in team_opponents_8gw[away_team_id]:
            d = team_opponents_8gw[away_team_id][event_id]
            d['opponent'].append(teams[home_team_id])
            d['difficulty'].append(away_diff)
            d['home_away'].append(f"{teams[away_team_id]}(A)")
            d['fixture_count'] += 1

print(f"   ✅ Fixtures indexed: {len([f for f in fixtures if f.get('finished')])} finished, "
      f"{len([f for f in fixtures if not f.get('finished') and f.get('event')])} upcoming")

# ---------------------------------------------------------------------------
# 4. LIVE GW DATA (xG, xGI for current and previous GW)
# ---------------------------------------------------------------------------

def fetch_gameweek_live(gameweek: int) -> dict:
    if gameweek and gameweek >= 1:
        url = f'https://fantasy.premierleague.com/api/event/{gameweek}/live/'
        print(f"   📡 Fetching GW{gameweek} live data...")
        result = fetch_fpl_data(url)
        return result if result else {}
    return {}


def extract_stat(gw_data: dict, stat_key: str) -> dict:
    """Extract a single stat keyed by player_id from live GW data."""
    if not gw_data or 'elements' not in gw_data:
        return {}
    return {
        el['id']: float(el['stats'].get(stat_key, 0) or 0)
        for el in gw_data['elements']
    }


current_gw_data  = fetch_gameweek_live(current_gameweek)
previous_gw_data = fetch_gameweek_live(current_gameweek - 1 if current_gameweek > 1 else None)

current_gw_xg   = extract_stat(current_gw_data,  'expected_goals')
previous_gw_xg  = extract_stat(previous_gw_data, 'expected_goals')
current_gw_xgi  = extract_stat(current_gw_data,  'expected_goal_involvements')
previous_gw_xgi = extract_stat(previous_gw_data, 'expected_goal_involvements')

# ---------------------------------------------------------------------------
# 5. BUILD PLAYER DATAFRAME
# ---------------------------------------------------------------------------

print("\n🔨 Building player DataFrame...")
player_info = []

for player in players_raw:
    pid = player['id']

    player_data = {
        # Identity
        'Photo':            player['photo'],
        'Player ID':        pid,
        'Player Name':      player['web_name'],
        'First Name':       player['first_name'],
        'Last Name':        player['second_name'],
        'Team':             teams[player['team']],
        'Position':         positions[player['element_type']],
        'Availability':     player['status'],
        'Removed':          player.get('removed', False),   # New field this season — True for players no longer in the PL
        'Current Gameweek': current_gameweek,

        # Pricing
        'Cost':             player['now_cost'] / 10,
        'Cost Change/GW':   player['cost_change_event'],

        # Performance
        'Form':             player['form'],
        'GW Points':        player['event_points'],
        'Total Points':     player['total_points'],
        'Points/Game':      player['points_per_game'],
        'Expected points Current GW': player['ep_this'],
        'Expected points Next GW':    player['ep_next'],

        # Attacking
        'Goals':            float(player['goals_scored']),
        'Assists':          float(player['assists']),
        'XG':               float(player['expected_goals']),
        'XA':               float(player['expected_assists']),
        'XGI':              float(player['expected_goal_involvements']),
        'XG/90':            float(player['expected_goals_per_90']),
        'XA/90':            float(player['expected_assists_per_90']),
        'XGI/90':           float(player['expected_goal_involvements_per_90']),

        # Defensive
        'Clean Sheets':     player['clean_sheets'],
        'Clean Sheets/90':  player['clean_sheets_per_90'],
        'Goals Conceded':   player['goals_conceded'],
        'Goals Conceded/90':player['goals_conceded_per_90'],
        'XGC':              float(player['expected_goals_conceded']),
        'XGC/90':           float(player['expected_goals_conceded_per_90']),
        'Saves':            player['saves'],
        'Saves/90':         player['saves_per_90'],
        'Penalties Saved':  player['penalties_saved'],

        # Disciplinary
        'Yellow Cards':     player['yellow_cards'],
        'Red Cards':        player['red_cards'],
        'Penalties Missed': player['penalties_missed'],

        # Playing time
        'Minutes':          player['minutes'],
        'Starts':           player['starts'],
        'Starts/90':        player['starts_per_90'],

        # Bonus / ICT
        'Total Bonus Point':        player['bonus'],
        'BPS':              player['bps'],
        'Influence':        player['influence'],
        'Creativity':       player['creativity'],
        'Threat':           player['threat'],
        'ICT Index':        player['ict_index'],
        'ICT Index Rank':   player['ict_index_rank'],

        # Ownership / transfers
        'Ownership (%)':    player['selected_by_percent'],
        'Transfers In':     player['transfers_in'],
        'Transfers Out':    player['transfers_out'],
        'GW Transfers In':  player['transfers_in_event'],
        'GW Transfers Out': player['transfers_out_event'],

        # Set pieces
        'Penalty Order':                    player['penalties_order'],
        'Freekick/Cornerkick Order':        player['corners_and_indirect_freekicks_order'],
        'Chance of playing next':           player['chance_of_playing_next_round'],

        # Dream team
        'In Dream Team':    player['in_dreamteam'],
        'DreamTeam Count':  player['dreamteam_count'],

        # Rankings
        'Form Rank':                player['form_rank'],
        'Form Rank/Position':       player['form_rank_type'],
        'Points/Game Rank':         player['points_per_game_rank'],
        'Position Ranking':         player['points_per_game_rank_type'],

        # Defensive contributions (newer API field)
        'Defensive Contributions':      player.get('defensive_contribution', 0),
        'Defensive Contributions/90':   player.get('defensive_contribution_per_90', 0),
    }

    # Next 5 GW fixture columns
    for gw in next_5_gameweeks:
        opp_info = team_opponents[player['team']][gw['id']]
        opponents_with_venue = [
            f"{opp}{venue[-3:]}"
            for opp, venue in zip(opp_info['opponent'], opp_info['home_away'])
        ]
        player_data[f'{gw["name"]}']             = ', '.join(opponents_with_venue)
        player_data[f'{gw["name"]} Difficulty']  = ', '.join(map(str, opp_info['difficulty']))

    # GW-level xG / xGI deltas
    xg_curr  = float(current_gw_xg.get(pid, 0))
    xg_prev  = float(previous_gw_xg.get(pid, 0))
    xgi_curr = float(current_gw_xgi.get(pid, 0))
    xgi_prev = float(previous_gw_xgi.get(pid, 0))

    player_data['XG Current GW']  = xg_curr
    player_data['XG Previous GW'] = xg_prev
    player_data['ΔG_GW']          = round(xg_curr - xg_prev, 3)

    player_data['XGI Current GW']  = xgi_curr
    player_data['XGI Previous GW'] = xgi_prev
    player_data['ΔGI']             = round(xgi_curr - xgi_prev, 3)

    # Derived metrics
    gi                       = player_data['Goals'] + player_data['Assists']
    player_data['GI']        = gi
    player_data['Delta G']   = round(player_data['Goals'] - player_data['XG'], 3)
    player_data['Delta GI']  = round(gi - player_data['XGI'], 3)

    player_info.append(player_data)

player_df = pd.DataFrame(player_info)
print(f"   ✅ Player DataFrame built: {len(player_df)} players, {len(player_df.columns)} columns")

# ---------------------------------------------------------------------------
# 6. ML PREDICTIONS
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("🤖 RUNNING ML PREDICTIONS")
print("="*70)

try:
    retrain_model = os.getenv("RETRAIN_MODEL", "false").lower() == "true"

    player_df, ml_model = add_ml_predictions_v2(
        player_df,
        teams_data,
        fixtures_data,
        retrain=retrain_model,
    )
    ml_success = True
    print(f"\n✅ ML predictions complete.")

except Exception as e:
    print(f"\n⚠️  ML prediction failed: {e}")
    print("   Falling back to Form-based xP approximation...")
    ml_success = False

    player_df['xP']            = pd.to_numeric(player_df['Form'], errors='coerce').fillna(0)
    player_df['xP_confidence'] = 0.0
    player_df['AI_Rating']     = 'N/A'

# ---------------------------------------------------------------------------
# 7. NORMALISE DYNAMIC GW COLUMNS
# Rename 'Gameweek N' → 'Next GW Opponent N' / 'Next GW Difficulty N'
# ---------------------------------------------------------------------------

difficulty_cols = sorted(
    [c for c in player_df.columns if re.match(r'Gameweek\s+\d+\s+Difficulty$', c)],
    key=gw_col_number
)
opponent_cols = sorted(
    [c for c in player_df.columns if re.match(r'Gameweek\s+\d+$', c)],
    key=gw_col_number
)

for i, col in enumerate(difficulty_cols, start=1):
    player_df.rename(columns={col: f'Next GW Difficulty {i}'}, inplace=True)
for i, col in enumerate(opponent_cols, start=1):
    player_df.rename(columns={col: f'Next GW Opponent {i}'}, inplace=True)

max_next = max(len(next_5_gameweeks), 1)

for i in range(1, 6):
    if f'Next GW Difficulty {i}' not in player_df.columns:
        player_df[f'Next GW Difficulty {i}'] = np.nan
    if f'Next GW Opponent {i}' not in player_df.columns:
        player_df[f'Next GW Opponent {i}'] = ''

player_df['Next GW Opponent']   = player_df['Next GW Opponent 1']
player_df['Next GW Difficulty'] = player_df['Next GW Difficulty 1'].apply(parse_difficulty)

# ---------------------------------------------------------------------------
# 8. FIXTURE DIFFICULTY METRICS
# ---------------------------------------------------------------------------

player_df['Difficulty Score'] = player_df.apply(
    lambda row: sum(
        parse_difficulty(row.get(f'Next GW Difficulty {i}', 0))
        for i in range(1, max_next + 1)
    ),
    axis=1,
)

player_df['FD Index'] = player_df.apply(
    lambda row: round(float(row['Form']) / row['Difficulty Score'], 3)
    if row['Difficulty Score'] not in (0, np.nan) and float(row['Form'] or 0) > 0
    else 0.0,
    axis=1,
)

player_df['Next 5 GW FDR'] = player_df.apply(
    lambda row: [
        parse_difficulty(row.get(f'Next GW Difficulty {i}', np.nan))
        for i in range(1, max_next + 1)
    ],
    axis=1,
)

# ---------------------------------------------------------------------------
# 8b. FDR HEATMAP DATA
# Builds the Fixture Planner sheet — one row per team, one col per GW (next 8).
# Uses exact FPL colour codes for 1–5 difficulty.
# Detects DGW (fixture_count > 1) and BGW (fixture_count == 0) automatically.
# ---------------------------------------------------------------------------

print("\n🔨 Building Fixture Planner (FDR heatmap)...")

fixture_planner_rows = []

for team_id, team_short in teams.items():
    row = {
        'Team':        team_short,
        'Team ID':     team_id,
    }

    avg_fdrs     = []
    dgw_count    = 0
    bgw_count    = 0

    for gw in next_8_gameweeks:
        gw_id   = gw['id']
        gw_name = gw['name']

        if team_id not in team_opponents_8gw or gw_id not in team_opponents_8gw[team_id]:
            # No fixture data — treat as BGW
            row[f'{gw_name} Opponent']    = 'BGW'
            row[f'{gw_name} FDR']         = 0
            row[f'{gw_name} FDR Colour']  = FDR_COLOURS['BGW']
            row[f'{gw_name} Type']        = 'BGW'
            row[f'{gw_name} Display']     = 'BGW —'
            bgw_count += 1
            continue

        gw_info = team_opponents_8gw[team_id][gw_id]
        count   = gw_info['fixture_count']

        if count == 0:
            # BGW — no fixture scheduled
            row[f'{gw_name} Opponent']    = 'BGW'
            row[f'{gw_name} FDR']         = 0
            row[f'{gw_name} FDR Colour']  = FDR_COLOURS['BGW']
            row[f'{gw_name} Type']        = 'BGW'
            row[f'{gw_name} Display']     = 'BGW —'
            bgw_count += 1

        elif count >= 2:
            # DGW — two fixtures in same GW
            # Use the average of both fixture difficulties
            valid_diffs = [d for d in gw_info['difficulty'] if d]
            avg_diff    = sum(valid_diffs) / len(valid_diffs) if valid_diffs else 3
            opps        = ', '.join([
                f"{opp}{ha[-3:]}"
                for opp, ha in zip(gw_info['opponent'], gw_info['home_away'])
            ])
            row[f'{gw_name} Opponent']    = opps
            row[f'{gw_name} FDR']         = round(avg_diff, 1)
            row[f'{gw_name} FDR Colour']  = FDR_COLOURS['DGW']
            row[f'{gw_name} Type']        = 'DGW'
            row[f'{gw_name} Display']     = f'DGW ⚡ {opps}'
            avg_fdrs.append(avg_diff)
            dgw_count += 1

        else:
            # Normal single fixture
            diff = gw_info['difficulty'][0] if gw_info['difficulty'] else 3
            opp  = gw_info['opponent'][0]   if gw_info['opponent']   else '?'
            ha   = gw_info['home_away'][0]  if gw_info['home_away']  else ''
            display = f"{opp}{ha[-3:]}"

            row[f'{gw_name} Opponent']    = display
            row[f'{gw_name} FDR']         = int(diff)
            row[f'{gw_name} FDR Colour']  = fdr_hex(diff)
            row[f'{gw_name} Type']        = 'Normal'
            row[f'{gw_name} Display']     = display
            avg_fdrs.append(float(diff))

    # Summary metrics per team
    row['Avg FDR Next 8 GW']   = round(sum(avg_fdrs) / len(avg_fdrs), 2) if avg_fdrs else 0
    row['Avg FDR Next 5 GW']   = round(sum(avg_fdrs[:5]) / len(avg_fdrs[:5]), 2) if avg_fdrs[:5] else 0
    row['Avg FDR Next 3 GW']   = round(sum(avg_fdrs[:3]) / len(avg_fdrs[:3]), 2) if avg_fdrs[:3] else 0
    row['DGW Count']            = dgw_count
    row['BGW Count']            = bgw_count
    row['Fixture Rating']       = (
        'Excellent'  if row['Avg FDR Next 5 GW'] <= 2.0 else
        'Good'       if row['Avg FDR Next 5 GW'] <= 2.8 else
        'Average'    if row['Avg FDR Next 5 GW'] <= 3.5 else
        'Difficult'
    )
    row['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

    fixture_planner_rows.append(row)

fixture_planner_df = pd.DataFrame(fixture_planner_rows).sort_values(
    by='Avg FDR Next 5 GW', ascending=True
).reset_index(drop=True)

print(f"   ✅ Fixture Planner built: {len(fixture_planner_df)} teams × {len(next_8_gameweeks)} GWs")

# ---------------------------------------------------------------------------
# 9. DASHBOARD-READY DERIVED COLUMNS
# ---------------------------------------------------------------------------

player_df['Net GW Transfers'] = player_df['GW Transfers In'] - player_df['GW Transfers Out']

player_df['Price Rise Score'] = player_df.apply(
    lambda row: round(
        (row['GW Transfers In'] - row['GW Transfers Out']) /
        max(row['GW Transfers In'] + row['GW Transfers Out'], 1),
        3
    ),
    axis=1,
)

player_df['Differential Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) *
        (1 - float(row.get('Ownership (%)', 0) or 0) / 100) * 10,
        2
    ),
    axis=1,
)

player_df['Captaincy Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) * 0.50 +
        float(row.get('Form', 0) or 0) * 0.30 +
        (5.0 - parse_difficulty(row.get('Next GW Difficulty 1', 3))) * 0.20,
        2
    ),
    axis=1,
)

player_df['Transfer In Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) * 0.40 +
        (5.0 - parse_difficulty(row.get('Next GW Difficulty 1', 3))) * 0.40 +
        float(row.get('Points/Game', 0) or 0) * 0.20,
        2
    ),
    axis=1,
)

player_df['Points Per Million'] = player_df.apply(
    lambda row: round(
        float(row.get('Total Points', 0) or 0) / max(float(row.get('Cost', 4.0) or 4.0), 1.0),
        2
    ),
    axis=1,
)

# DGW / BGW flags — derived from Next GW Opponent 1 (comma = two opponents = DGW)
player_df['Has DGW'] = player_df['Next GW Opponent 1'].apply(
    lambda x: '⚡ DGW' if (isinstance(x, str) and ',' in x) else ''
)
player_df['Has BGW'] = player_df['Next GW Opponent 1'].apply(
    lambda x: '⚠️ BGW' if (not isinstance(x, str) or x.strip() == '') else ''
)

# GW average — for xP vs Average comparison column on player level
player_df['GW Average Points'] = gw_averages.get(current_gameweek, 0)
player_df['GW Points vs Average'] = player_df['GW Points'] - player_df['GW Average Points']

player_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

# ---------------------------------------------------------------------------
# 10. POSITION-FILTERED SMART PICKS TABLES
# ---------------------------------------------------------------------------

def create_smart_picks_table(position_name: str, top_n: int) -> pd.DataFrame:
    """
    Build a smart picks table for a given position.
    Sorted by GW Transfers In (market momentum) but with all ML columns included.
    """
    base_cols = [
        # Identity
        'Player Name', 'Availability', 'Team', 'Position', 'Cost',
        # ML predictions
        'xP', 'xP_confidence', 'AI_Rating',
        # Composite scores
        'Captaincy Score', 'Transfer In Score', 'Differential Score',
        # Core metrics
        'Form', 'FD Index', 'Points/Game', 'Total Points',
        'GW Points', 'Points Per Million',
        # xG family
        'XG', 'XA', 'XGI', 'XG/90', 'XA/90', 'XGI/90',
        'XG Current GW', 'XG Previous GW', 'ΔG_GW',
        'Delta G', 'Delta GI', 'ΔGI',
        # Attacking / defensive
        'Goals', 'Assists', 'GI',
        'Clean Sheets', 'Saves', 'Defensive Contributions',
        # Playing time
        'Minutes', 'Starts',
        # Discipline
        'Yellow Cards', 'Red Cards',
        # Ownership
        'Ownership (%)', 'GW Transfers In', 'GW Transfers Out',
        'Net GW Transfers', 'Price Rise Score',
        # Fixture
        'Next GW Opponent', 'Next GW Difficulty', 'Difficulty Score',
        'Has DGW', 'Has BGW',
        # GW average comparison
        'GW Average Points', 'GW Points vs Average',
        # Meta
        'Expected points Next GW', 'Current Gameweek', 'Last Updated',
    ]

    next_gw_cols = []
    for i in range(1, max_next + 1):
        next_gw_cols.append(f'Next GW Opponent {i}')
        next_gw_cols.append(f'Next GW Difficulty {i}')

    all_cols   = base_cols + next_gw_cols
    df_pos     = player_df[player_df['Position'] == position_name].copy()
    final_cols = [c for c in all_cols if c in df_pos.columns]

    return (
        df_pos[final_cols]
        .sort_values(by='GW Transfers In', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


goalkeepers_smart_picks  = create_smart_picks_table('Goalkeeper', top_n=5)
defenders_smart_picks    = create_smart_picks_table('Defender',   top_n=10)
midfielders_smart_picks  = create_smart_picks_table('Midfielder', top_n=15)
forwards_smart_picks     = create_smart_picks_table('Forward',    top_n=5)

# ---------------------------------------------------------------------------
# 11. TEAM STATS TABLES
# ---------------------------------------------------------------------------

print("\n🔨 Building team stats tables...")

teams_bootstrap_raw = data['teams']
teams_df = pd.DataFrame(teams_bootstrap_raw)[
    ['id', 'short_name', 'name', 'played', 'points', 'form']
].rename(columns={'short_name': 'team', 'name': 'Team'})

teams_df['Goals Scored']   = 0
teams_df['Goals Conceded'] = 0
teams_df['Games']          = 0

team_results = {team_id: [] for team_id in teams_df['id']}

for fixture in fixtures:
    if not fixture.get('finished'):
        continue

    h_id    = fixture.get('team_h')
    a_id    = fixture.get('team_a')
    h_goals = fixture.get('team_h_score') or 0
    a_goals = fixture.get('team_a_score') or 0

    if not (h_id and a_id):
        continue

    teams_df.loc[teams_df['id'] == h_id, 'Goals Scored']   += h_goals
    teams_df.loc[teams_df['id'] == h_id, 'Goals Conceded'] += a_goals
    teams_df.loc[teams_df['id'] == a_id, 'Goals Scored']   += a_goals
    teams_df.loc[teams_df['id'] == a_id, 'Goals Conceded'] += h_goals
    teams_df.loc[teams_df['id'] == h_id, 'Games'] += 1
    teams_df.loc[teams_df['id'] == a_id, 'Games'] += 1

    h_short    = teams_df.loc[teams_df['id'] == h_id, 'team'].values[0]
    a_short    = teams_df.loc[teams_df['id'] == a_id, 'team'].values[0]
    result_str = f"{h_short} {h_goals} - {a_goals} {a_short}"
    team_results[h_id].append(result_str)
    team_results[a_id].append(result_str)

teams_df['Last 5 GW Results'] = teams_df['id'].apply(
    lambda tid: ', '.join(team_results[tid][-5:])
)
teams_df['Goals Scored/Game']   = (teams_df['Goals Scored']   / teams_df['Games'].replace(0, np.nan)).round(2)
teams_df['Goals Conceded/Game'] = (teams_df['Goals Conceded'] / teams_df['Games'].replace(0, np.nan)).round(2)
teams_df['Goal Difference']     = teams_df['Goals Scored'] - teams_df['Goals Conceded']
teams_df['Last Updated']        = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

attacking_teams = teams_df.sort_values(
    by=['Goals Scored/Game', 'Goals Scored', 'Goal Difference'],
    ascending=[False, False, False]
)[['Team', 'Games', 'Goals Scored/Game', 'Goals Conceded/Game', 'Goals Scored',
   'Goals Conceded', 'Goal Difference', 'Last 5 GW Results', 'Last Updated']]

defensive_teams = teams_df.sort_values(
    by=['Goals Conceded/Game', 'Goals Conceded'],
    ascending=[True, True]
)[['Team', 'Games', 'Goals Conceded/Game', 'Goals Conceded', 'Goals Scored/Game',
   'Goals Scored', 'Goal Difference', 'Last 5 GW Results', 'Last Updated']]

# ---------------------------------------------------------------------------
# 12. KEY METRICS GUIDE
# ---------------------------------------------------------------------------

status_code_info = [
    ["Status Code", "Meaning", "FPL Action"],
    ["'a'", "Available — Fully fit",               "Start, transfer in freely"],
    ["'d'", "Doubtful — Minor injury risk",         "Check pre-deadline news"],
    ["'i'", "Injured — Not available",             "Sell or bench immediately"],
    ["'s'", "Suspended — Banned",                  "Avoid until ban served"],
    ["'u'", "Unavailable — Non-injury reason",     "Avoid for this GW"],
    ["'n'", "Not in squad — Rotation risk",        "Sell unless price holds value"],
    [],
    ["Metric",             "Definition",                                      "FPL Usage Tip"],
    ["xP",                 "ML-predicted points next GW (ensemble model)",    "Primary pick signal — trust over Form"],
    ["xP_confidence",      "Prediction uncertainty (std dev across models)",  "Low = high confidence. High = risky"],
    ["AI_Rating",          "Premium / Good / Average / Monitor / Avoid",      "Quick visual filter for transfers"],
    ["Captaincy Score",    "xP×0.5 + Form×0.3 + Fixture×0.2",               "Sort descending for captain pick"],
    ["Transfer In Score",  "xP×0.4 + Fixture×0.4 + PPG×0.2",                "Sort descending for who to buy"],
    ["Differential Score", "xP × (1 - Ownership%) × 10",                    "High = rank-gaining differentials"],
    ["FD Index",           "Form ÷ Difficulty Score",                         "In-form players in easy fixtures"],
    ["XG",                 "Expected Goals (season)",                          "Goal threat indicator"],
    ["Delta G",            "Goals − xG",                                       "+ve = clinical, −ve = may regress"],
    ["XA",                 "Expected Assists",                                 "Creative involvement proxy"],
    ["Delta GI",           "GI − (xG + xA)",                                  "+ve = outperforming"],
    ["ΔG_GW",              "XG Current GW − XG Previous GW",                 "+ve = improving momentum"],
    ["ΔGI",                "XGI Current GW − XGI Previous GW",               "Weekly momentum tracker"],
    ["Points Per Million", "Total Points ÷ Cost",                             "Best value in price range"],
    ["Net GW Transfers",   "GW Transfers In − GW Transfers Out",             "+ve = market buying"],
    ["Price Rise Score",   "Normalised net transfer direction (−1 to +1)",    "+0.3+ → price rise likely"],
    ["GW Points vs Avg",   "GW Points − GW Average",                          "+ve = beat the field this GW"],
    [],
    ["FDR Colour",  "FPL Official Colour",       "Meaning"],
    ["1",           "#375523 Dark Green",         "Very Easy fixture"],
    ["2",           "#01FC7A Bright Green",        "Easy fixture"],
    ["3",           "#E7E7E7 Light Grey",          "Medium fixture"],
    ["4",           "#FF1751 Red",                 "Hard fixture"],
    ["5",           "#80072D Dark Red",            "Very Hard fixture"],
    ["DGW",         "#DAA520 Gold",                "Double Gameweek — team plays twice"],
    ["BGW",         "#4A4A4A Dark Grey",           "Blank Gameweek — team doesn't play"],
]

status_code_df = pd.DataFrame(status_code_info)
status_code_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

# ---------------------------------------------------------------------------
# 13. MY FPL — PERSONAL TEAM DATA (Team ID: 2551369)
# Feeds the My Season page of the dashboard.
# All sections wrapped in try/except — personal API failure never kills pipeline.
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print(f"👤 FETCHING PERSONAL FPL DATA (Team ID: {MY_TEAM_ID})")
print("="*70)

player_lookup = {p['id']: p for p in players_raw}

# ── 13a. SEASON HISTORY ───────────────────────────────────────────────────
# One row per completed GW.
# Now includes KPI delta columns needed by dashboard:
#   Rank Change, Rank Direction (↑/↓/→), Points vs Average,
#   Points vs TOTW (Team of the Week), cumulative total.
#
# history_fetch_ok distinguishes "the API call failed" from "the API call
# succeeded and correctly says there's no data yet" (e.g. preseason, before
# GW1). Those used to be treated identically — both left this sheet
# untouched — which meant every preseason the sheet (and therefore the
# dashboard's headline numbers) stayed frozen on last season's final
# stats. Now a genuine "nothing yet" response overwrites that stale data
# with an honest, empty, up-to-date sheet instead.

my_season_history_df = pd.DataFrame()
my_past_seasons_df   = pd.DataFrame()
history_fetch_ok     = False

try:
    print(f"\n   📡 Fetching season history...")
    history_data = fetch_fpl_data(
        f'https://fantasy.premierleague.com/api/entry/{MY_TEAM_ID}/history/'
    )

    if history_data is not None:
        history_fetch_ok = True
        gw_history = history_data.get('current', [])

        chips_used = {
            chip.get('event'): chip.get('name', '').replace('_', ' ').title()
            for chip in history_data.get('chips', [])
        }

        season_rows   = []
        running_total = 0

        for gw in gw_history:
            gw_num  = gw.get('event')
            gw_pts  = gw.get('points', 0)
            hits    = gw.get('event_transfers_cost', 0)
            net_pts = gw_pts - hits
            running_total += net_pts

            gw_avg  = gw_averages.get(gw_num, 0)
            gw_high = gw_highest.get(gw_num, 0)
            vs_avg  = gw_pts - gw_avg
            vs_high = gw_pts - gw_high

            season_rows.append({
                'Gameweek':              gw_num,
                'GW Points':             gw_pts,
                'Transfers Cost (Hits)': hits,
                'Net GW Points':         net_pts,
                'Cumulative Points':     running_total,
                'Overall Rank':          gw.get('overall_rank'),
                'GW Rank':               gw.get('rank'),
                'Team Value':            round(gw.get('value', 0) / 10, 1),
                'Money In Bank':         round(gw.get('bank', 0) / 10, 1),
                'Total Transfers Made':  gw.get('event_transfers', 0),
                'Hit Taken':             'Yes ❌' if hits > 0 else 'No ✅',
                'Points On Bench':       gw.get('points_on_bench', 0),
                'GW Average':            gw_avg,
                'GW Highest Score':      gw_high,
                'Points vs Average':     vs_avg,
                'Points vs TOTW':        vs_high,
                'Beat Average':          'Yes ✅' if vs_avg > 0 else 'No ❌',
                'Chip Used':             chips_used.get(gw_num, ''),
            })

        my_season_history_df = pd.DataFrame(season_rows)

        if not my_season_history_df.empty:
            # Rank change (positive = moved UP — improved rank number is smaller)
            my_season_history_df['Rank Change'] = (
                my_season_history_df['Overall Rank'].shift(1) -
                my_season_history_df['Overall Rank']
            ).fillna(0).astype(int)

            # Rank direction arrow — used by dashboard KPI card
            my_season_history_df['Rank Direction'] = my_season_history_df['Rank Change'].apply(
                lambda x: '↑ Improved' if x > 0 else ('↓ Dropped' if x < 0 else '→ Held')
            )

            # Rank delta display string (e.g. "↑ +18,240" or "↓ -284,476")
            my_season_history_df['Rank Delta Display'] = my_season_history_df['Rank Change'].apply(
                lambda x: f"↑ +{x:,}" if x > 0 else (f"↓ {x:,}" if x < 0 else '→ No change')
            )

            # Points delta display string (e.g. "+73 pts this GW")
            # Leading apostrophe forces Sheets to treat this as literal text —
            # without it, a cell value starting with "+" gets auto-parsed as
            # the start of a formula and displays "#ERROR!" instead of the text.
            my_season_history_df['Points Delta Display'] = my_season_history_df['GW Points'].apply(
                lambda x: f"'+{x} pts this GW"
            )

            # Beat average string — same leading-character issue when positive
            my_season_history_df['vs Average Display'] = my_season_history_df['Points vs Average'].apply(
                lambda x: f"'+{x} vs avg" if x > 0 else f"{x} vs avg"
            )

            my_season_history_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

            print(f"   ✅ Season history: {len(my_season_history_df)} gameweeks")
            print(f"      Total points : {running_total} | "
                  f"Best rank: #{my_season_history_df['Overall Rank'].min():,}")
        else:
            print("   ℹ️  No completed gameweeks yet this season — writing an empty, up-to-date history sheet.")

        # Real past-season summaries straight from the FPL API (season name,
        # points, rank). This replaces any hand-maintained/hardcoded past
        # seasons list on the dashboard side — it's always accurate and
        # never needs a manual edit when a season rolls over.
        past_rows = [
            {
                'Season':       ps.get('season_name', ''),
                'Total Points': ps.get('total_points', 0),
                'Overall Rank': ps.get('rank'),
            }
            for ps in history_data.get('past', [])
        ]
        my_past_seasons_df = pd.DataFrame(past_rows)
        if not my_past_seasons_df.empty:
            my_past_seasons_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')
        print(f"   ✅ Past seasons on record: {len(my_past_seasons_df)}")

    else:
        print("   ⚠️  Could not fetch season history (no response after retries) — leaving existing sheet untouched.")

except Exception as e:
    print(f"   ❌ Season history fetch failed: {e}")

# ── 13b. CURRENT GW SQUAD ─────────────────────────────────────────────────
# 15 players this GW — starting XI, bench, captain, VC.
# Merged with ML predictions and fixture data from the main pipeline.
# Free transfer logic correctly handles hits and the 1-FT-carry-over rule.
# squad_fetch_ok follows the same "failed vs genuinely empty" reasoning as
# 13a. squad_team_value / squad_bank are captured here (straight from
# entry_history, which FPL populates even before GW1) so the KPI section
# below has a real number to show for Team Value during preseason instead
# of a placeholder.

my_current_squad_df = pd.DataFrame()
squad_fetch_ok       = False
squad_team_value     = 0.0
squad_bank           = 0.0

try:
    print(f"\n   📡 Fetching GW{current_gameweek} squad picks...")
    current_squad_data = fetch_fpl_data(
        f'https://fantasy.premierleague.com/api/entry/{MY_TEAM_ID}/event/{current_gameweek}/picks/'
    )

    if current_squad_data is not None:
        squad_fetch_ok = True
        picks         = current_squad_data.get('picks', [])
        active_chip   = current_squad_data.get('active_chip') or ''
        entry_history = current_squad_data.get('entry_history', {})
        squad_team_value = round(entry_history.get('value', 0) / 10, 1)
        squad_bank       = round(entry_history.get('bank', 0) / 10, 1)

        # Correct FPL free transfer calculation:
        # - 1 free transfer per GW by default, carries over to max 2
        # - Any transfers beyond free allowance cost 4pts each (hit)
        # - If hit was taken, we know transfers > free allowance
        transfers_made = entry_history.get('event_transfers', 0)
        hit_cost       = entry_history.get('event_transfers_cost', 0)

        # Free transfers remaining next GW:
        # If 0 transfers this GW → banked 1 FT → 2 next GW
        # If 1 transfer (no hit) → used the FT → 1 next GW
        # If hit taken → still get 1 FT next GW
        if transfers_made == 0:
            free_transfers_next = 2   # banked
        elif hit_cost == 0:
            free_transfers_next = max(0, 2 - transfers_made)
        else:
            free_transfers_next = 1   # always get 1 after a hit GW

        # xP for starting XI — from the main pipeline (populated after merge below)
        squad_rows = []

        for pick in picks:
            pid          = pick['element']
            multiplier   = pick['multiplier']
            is_captain   = pick['is_captain']
            is_vice      = pick['is_vice_captain']
            position_num = pick['position']

            info = player_lookup.get(pid, {})
            pos  = positions.get(info.get('element_type'), 'Unknown')
            team = teams.get(info.get('team'), 'Unknown')

            if is_captain:
                captain_label = '👑 C' if multiplier == 2 else '👑 TC'
            elif is_vice:
                captain_label = 'VC'
            else:
                captain_label = ''

            squad_rows.append({
                'Position Slot':           position_num,
                'Player Name':             info.get('web_name', 'Unknown'),
                'Full Name':               f"{info.get('first_name', '')} {info.get('second_name', '')}".strip(),
                'Team':                    team,
                'Position':                pos,
                'Is Starting':             'Yes' if position_num <= 11 else 'No (Bench)',
                'Captain':                 captain_label,
                'Multiplier':              multiplier,
                'Cost':                    round(info.get('now_cost', 0) / 10, 1),
                'Form':                    info.get('form', 0),
                'GW Points':               info.get('event_points', 0),
                'GW Points x Multiplier':  info.get('event_points', 0) * max(multiplier, 1),
                'Total Points':            info.get('total_points', 0),
                'Ownership (%)':           info.get('selected_by_percent', 0),
                'Status':                  info.get('status', 'a'),
                # Filled from player_df merge below
                'xP':                      0.0,
                'xP Next GW':              0.0,
                'AI_Rating':               'N/A',
                'Captaincy Score':         0.0,
                'Next GW Opponent 1':      '',
                'Next GW Difficulty 1':    '',
                'FD Index':                0.0,
                'Delta GI':                0.0,
                'ΔGI':                     0.0,
                # Context
                'Active Chip':             active_chip,
                'Free Transfers Next GW':  free_transfers_next,
                'Hit Cost This GW':        hit_cost,
                'Transfers Made This GW':  transfers_made,
                'Last Updated':            pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC'),
            })

        my_current_squad_df = pd.DataFrame(squad_rows)

        if not my_current_squad_df.empty:
            # Merge ML predictions + fixture data from the main player_df
            ml_merge_cols = [
                'Player Name', 'xP', 'xP_confidence', 'AI_Rating',
                'Captaincy Score', 'Differential Score', 'Transfer In Score',
                'Next GW Opponent 1', 'Next GW Difficulty 1',
                'FD Index', 'Delta GI', 'ΔGI',
                'Expected points Next GW',
            ]
            available_ml_cols = [c for c in ml_merge_cols if c in player_df.columns]
            merge_source = player_df[available_ml_cols].drop_duplicates('Player Name')

            my_current_squad_df = my_current_squad_df.merge(
                merge_source,
                on='Player Name',
                how='left',
                suffixes=('', '_merged')
            )

            for col in ['xP', 'AI_Rating', 'Captaincy Score',
                        'Next GW Opponent 1', 'Next GW Difficulty 1',
                        'FD Index', 'Delta GI', 'ΔGI']:
                merged_col = f'{col}_merged'
                if merged_col in my_current_squad_df.columns:
                    my_current_squad_df[col] = my_current_squad_df[merged_col].fillna(
                        my_current_squad_df[col]
                    )
                    my_current_squad_df.drop(columns=[merged_col], inplace=True)

            # xP Next GW alias from Expected points Next GW
            if 'Expected points Next GW_merged' in my_current_squad_df.columns:
                my_current_squad_df['xP Next GW'] = my_current_squad_df['Expected points Next GW_merged']
                my_current_squad_df.drop(columns=['Expected points Next GW_merged'], inplace=True)
            elif 'Expected points Next GW' in my_current_squad_df.columns:
                my_current_squad_df['xP Next GW'] = my_current_squad_df['Expected points Next GW']

            # Summary stats
            starting_xi   = my_current_squad_df[my_current_squad_df['Is Starting'] == 'Yes']
            bench_players = my_current_squad_df[my_current_squad_df['Is Starting'] == 'No (Bench)']
            starting_xp   = starting_xi['xP'].sum()
            bench_xp      = bench_players['xP'].sum()
            total_gw_pts  = starting_xi['GW Points x Multiplier'].sum()

            print(f"   ✅ Current squad: {len(my_current_squad_df)} players")
            print(f"      Starting xP: {starting_xp:.1f} | Bench xP: {bench_xp:.1f}")
            print(f"      GW Points (with captain): {total_gw_pts} | Hit: -£{hit_cost}m")
            print(f"      Free transfers next GW: {free_transfers_next}")
            if active_chip:
                print(f"      🎯 Active chip: {active_chip}")
        else:
            print(f"   ℹ️  No squad picks saved yet for GW{current_gameweek} — writing an empty, up-to-date squad sheet.")

    else:
        print(f"   ⚠️  Could not fetch GW{current_gameweek} squad (no response after retries) — leaving existing sheet untouched.")

except Exception as e:
    print(f"   ❌ Current squad fetch failed: {e}")

# ── 13c. TRANSFER HISTORY ─────────────────────────────────────────────────

my_transfers_df    = pd.DataFrame()
transfers_fetch_ok = False

try:
    print(f"\n   📡 Fetching transfer history...")
    transfers_data = fetch_fpl_data(
        f'https://fantasy.premierleague.com/api/entry/{MY_TEAM_ID}/transfers/'
    )

    if transfers_data is not None:
        transfers_fetch_ok = True
        transfer_rows = []

        for t in transfers_data:
            pid_in   = t.get('element_in')
            pid_out  = t.get('element_out')
            info_in  = player_lookup.get(pid_in,  {})
            info_out = player_lookup.get(pid_out, {})

            cost_in   = round(t.get('element_in_cost',  0) / 10, 1)
            cost_out  = round(t.get('element_out_cost', 0) / 10, 1)
            cost_diff = round(cost_in - cost_out, 1)

            transfer_rows.append({
                'Gameweek':         t.get('event'),
                'Player In':        info_in.get('web_name', 'Unknown'),
                'Player In Team':   teams.get(info_in.get('team'), ''),
                'Player In Cost':   cost_in,
                'Player In Pos':    positions.get(info_in.get('element_type'), ''),
                'Player Out':       info_out.get('web_name', 'Unknown'),
                'Player Out Team':  teams.get(info_out.get('team'), ''),
                'Player Out Cost':  cost_out,
                'Player Out Pos':   positions.get(info_out.get('element_type'), ''),
                'Cost Difference':  cost_diff,
                'Transfer Time':    t.get('time', ''),
            })

        my_transfers_df = pd.DataFrame(transfer_rows)

        if not my_transfers_df.empty:
            my_transfers_df = my_transfers_df.sort_values(
                by=['Gameweek', 'Transfer Time'], ascending=[False, False]
            ).reset_index(drop=True)
            my_transfers_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')
            print(f"   ✅ Transfer history: {len(my_transfers_df)} transfers across the season")
        else:
            print("   ℹ️  No transfers made yet this season — writing an empty, up-to-date transfers sheet.")

    else:
        print("   ⚠️  Could not fetch transfer history (no response after retries) — leaving existing sheet untouched.")

except Exception as e:
    print(f"   ❌ Transfer history fetch failed: {e}")

# ── 13d. MY FPL KPI SUMMARY ───────────────────────────────────────────────
# Single-row table that the dashboard reads to populate the 5 KPI cards.
# This means the dashboard only needs to read ONE cell per KPI — no formulas.
# Columns: Total Points, Overall Rank, Rank Change, Rank Direction,
#          Team Value, Free Transfers Next GW, Hit Cost,
#          xP Starting XI, GW Average, GW Points, GW Points Delta Display,
#          Rank Delta Display, Active Chip, Season Label, Season Status,
#          Last Updated.
#
# THE CORE FIX: this used to only rebuild when BOTH history and squad had
# rows — which is precisely false every preseason. That meant the sheet
# (and therefore the dashboard's topbar season pill and all five headline
# KPI cards) stayed frozen on whatever was last written, i.e. last
# season's final numbers, for the entire close season. It now rebuilds
# whenever EITHER fetch succeeded, using honest zero/blank values for
# whichever half (history or squad) genuinely has nothing yet, tagged
# with a 'Season Status' the dashboard can use to show "Preseason"
# instead of pretending those zeros are real form.

my_kpi_df = pd.DataFrame()

try:
    if history_fetch_ok or squad_fetch_ok:

        has_history = not my_season_history_df.empty
        has_squad   = not my_current_squad_df.empty

        latest_gw = my_season_history_df.iloc[-1] if has_history else None

        # Current GW xP prediction (sum of starting XI) — available even
        # preseason, since a saved squad already has ML predictions merged in.
        starting_xi_xp = (
            my_current_squad_df[my_current_squad_df['Is Starting'] == 'Yes']['xP'].sum()
            if has_squad else 0.0
        )

        # xP vs GW average (using ML xP prediction, not actual — forward-looking)
        gw_avg_pts = gw_averages.get(current_gameweek, 0)
        xp_vs_avg  = round(starting_xi_xp - gw_avg_pts, 1)

        # Free transfers (from squad section)
        ft_next = (
            int(my_current_squad_df['Free Transfers Next GW'].iloc[0])
            if has_squad and 'Free Transfers Next GW' in my_current_squad_df.columns
            else 1
        )
        hit_cost_this_gw = (
            int(my_current_squad_df['Hit Cost This GW'].iloc[0])
            if has_squad and 'Hit Cost This GW' in my_current_squad_df.columns
            else 0
        )
        active_chip_this_gw = (
            str(my_current_squad_df['Active Chip'].iloc[0])
            if has_squad and 'Active Chip' in my_current_squad_df.columns
            else ''
        )

        kpi_row = {
            # Points KPI — honest 0s when no GW has been played yet this season
            'Total Points':              int(latest_gw['Cumulative Points']) if has_history else 0,
            'GW Points':                 int(latest_gw['GW Points']) if has_history else 0,
            'GW Points Delta Display':   latest_gw['Points Delta Display'] if has_history else 'Season not started',
            'Net GW Points':             int(latest_gw['Net GW Points']) if has_history else 0,

            # Rank KPI
            'Overall Rank':              int(latest_gw['Overall Rank']) if has_history else 0,
            'Rank Change':               int(latest_gw['Rank Change']) if has_history else 0,
            'Rank Direction':            latest_gw['Rank Direction'] if has_history else '—',
            'Rank Delta Display':        latest_gw['Rank Delta Display'] if has_history else '—',

            # Team value KPI — falls back to the live squad value/bank
            # (captured in 13b from entry_history) when there's no history yet
            'Team Value':                float(latest_gw['Team Value']) if has_history else squad_team_value,
            'Money In Bank':             float(latest_gw['Money In Bank']) if has_history else squad_bank,

            # xP vs Average KPI
            'xP Starting XI':            round(starting_xi_xp, 1),
            'GW Average':                gw_avg_pts,
            'xP vs Average':             xp_vs_avg,
            # Leading apostrophe forces literal text — see note in 13a above
            'xP vs Avg Display':         f"'+{xp_vs_avg} vs avg" if xp_vs_avg >= 0 else f"{xp_vs_avg} vs avg",

            # Free transfers KPI
            'Free Transfers Next GW':    ft_next,
            'Hit Cost This GW':          hit_cost_this_gw,
            'Hit Display':               f"'-{hit_cost_this_gw} pts hit" if hit_cost_this_gw > 0 else 'No hit taken',
            'Active Chip':               active_chip_this_gw,

            # Season context
            'Current Gameweek':          int(current_gameweek),
            'Season Label':              season_label,
            'Season Status':             'Active' if has_history else 'Preseason',
            'Season Best Rank':          int(my_season_history_df['Overall Rank'].min()) if has_history else 0,
            'Season Total Hits Cost':    int(my_season_history_df['Transfers Cost (Hits)'].sum()) if has_history else 0,
            'Season Points On Bench':    int(my_season_history_df['Points On Bench'].sum()) if has_history else 0,
            'GWs Beat Average':          int((my_season_history_df['Points vs Average'] > 0).sum()) if has_history else 0,

            'Last Updated':              pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC'),
        }

        my_kpi_df = pd.DataFrame([kpi_row])

        if has_history:
            print(f"\n   ✅ KPI Summary built for GW{current_gameweek} ({season_label}, Active)")
            print(f"      Points: {kpi_row['Total Points']} | Rank: #{kpi_row['Overall Rank']:,} "
                  f"| {kpi_row['Rank Delta Display']} | xP: {kpi_row['xP Starting XI']}")
        else:
            print(f"\n   ✅ KPI Summary built for GW{current_gameweek} ({season_label}, Preseason)")
            print(f"      Team Value: £{kpi_row['Team Value']}m | Starting XI xP: {kpi_row['xP Starting XI']}")

    else:
        print("   ⚠️  Skipping KPI summary — neither history nor squad fetch succeeded this run; leaving existing sheet untouched.")

except Exception as e:
    print(f"   ❌ KPI summary build failed: {e}")

# ---------------------------------------------------------------------------
# 14. WRITE ALL SHEETS
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("📝 WRITING ALL DATA TO GOOGLE SHEETS")
print("="*70)

# General sheets
write_to_sheet(sheet, player_df,               'Player Data')
write_to_sheet(sheet, goalkeepers_smart_picks,  'Smart Picks - Goalkeepers')
write_to_sheet(sheet, defenders_smart_picks,    'Smart Picks - Defenders')
write_to_sheet(sheet, midfielders_smart_picks,  'Smart Picks - Midfielders')
write_to_sheet(sheet, forwards_smart_picks,     'Smart Picks - Forwards')
write_to_sheet(sheet, attacking_teams,          'Best Attacking Teams')
write_to_sheet(sheet, defensive_teams,          'Best Defensive Teams')
write_to_sheet(sheet, fixture_planner_df,       'Fixture Planner')
write_to_sheet(sheet, status_code_df,           'FPL Key Metrics Guide')

# Personal sheets — gated on whether the fetch actually succeeded, not on
# whether the result happened to be empty. A real fetch failure preserves
# whatever is already in the sheet; a successful-but-empty response (e.g.
# preseason) correctly overwrites stale data with an honest empty state.
if history_fetch_ok:
    write_to_sheet(sheet, my_season_history_df,  'My FPL - Season History')
    write_to_sheet(sheet, my_past_seasons_df,     'My FPL - Past Seasons')

if squad_fetch_ok:
    write_to_sheet(sheet, my_current_squad_df,   'My FPL - Current Squad')

if transfers_fetch_ok:
    write_to_sheet(sheet, my_transfers_df,        'My FPL - Transfers')

if not my_kpi_df.empty:
    write_to_sheet(sheet, my_kpi_df,              'My FPL - KPI Summary')

# ---------------------------------------------------------------------------
# 15. PIPELINE SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("✅ FPL DATA PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"  📊 Players processed    : {len(player_df)}")
print(f"  🤖 ML predictions       : {'✅ Active' if ml_success else '⚠️  Fallback (Form)'}")
print(f"  📅 Fixture Planner      : {len(fixture_planner_df)} teams × {len(next_8_gameweeks)} GWs (FPL colours)")
print(f"  👤 Personal data        : Team ID {MY_TEAM_ID} · Season {season_label}")
print(f"     Season history       : {len(my_season_history_df)} GWs (written)" if history_fetch_ok else "     Season history       : ⚠️  Fetch failed — sheet left untouched")
print(f"     Past seasons         : {len(my_past_seasons_df)} seasons (written)" if history_fetch_ok else "     Past seasons         : ⚠️  Fetch failed — sheet left untouched")
print(f"     Current squad        : {len(my_current_squad_df)} players (written)" if squad_fetch_ok else "     Current squad        : ⚠️  Fetch failed — sheet left untouched")
print(f"     Transfer history     : {len(my_transfers_df)} transfers (written)" if transfers_fetch_ok else "     Transfer history     : ⚠️  Fetch failed — sheet left untouched")
print(f"     KPI Summary          : ✅ Written ({my_kpi_df.iloc[0]['Season Status']})" if not my_kpi_df.empty else "     KPI Summary          : ⚠️  Skipped")
print(f"  📈 New columns          : GW Points vs Average · Rank Delta Display · Points Delta Display")
print(f"                            Free Transfers Next GW · Hit Display · FDR Colours (FPL official)")
print(f"                            DGW/BGW auto-detection · Avg FDR Next 3/5/8 GW · Fixture Rating")
print(f"  📅 Gameweek             : GW{current_gameweek}")
print(f"  🔗 Sheet                : https://docs.google.com/spreadsheets/d/{sheet_id}")
print(f"  ⏱️  Completed            : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}")
print("="*70)
