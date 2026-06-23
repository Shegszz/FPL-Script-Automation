# =============================================================================
# fpl_update.py
# FPL Analytics Engine — Daily Data Pipeline
# Author: Segun Bakare | github.com/Shegszz/FPL-Script-Automation
#
# Pipeline:
#   FPL API → Python ETL → Google Sheets (auto-refreshed daily via GitHub Actions)
#
# Sheets written:
#   - Player Data              (all players, all metrics + ML predictions)
#   - Smart Picks - Goalkeepers
#   - Smart Picks - Defenders
#   - Smart Picks - Midfielders
#   - Smart Picks - Forwards
#   - Best Attacking Teams
#   - Best Defensive Teams
#   - Rank & Gameweek History  (for dashboard rank trajectory chart)
#   - FPL Key Metrics Guide
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
# HELPERS
# ---------------------------------------------------------------------------

def write_to_sheet(sheet, df: pd.DataFrame, sheet_name: str) -> None:
    """Write a DataFrame to a named worksheet, creating it if it doesn't exist."""
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="2000", cols="50")

    worksheet.clear()
    set_with_dataframe(worksheet, df, include_column_header=True, resize=True)
    worksheet.freeze(rows=1, cols=2)
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
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return 0.0


def gw_col_number(colname: str) -> int:
    """Extract integer from 'Gameweek N' column name for sorting."""
    m = re.search(r'Gameweek\s+(\d+)', colname)
    return int(m.group(1)) if m else 0

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

players_raw  = data['elements']
teams        = {team['id']: team['short_name'] for team in data['teams']}
positions    = {pos['id']: pos['singular_name'] for pos in data['element_types']}
events       = data['events']
teams_data   = data['teams']   # Full team list — passed to ML model

# ---------------------------------------------------------------------------
# 2. CURRENT GAMEWEEK + NEXT 5
# ---------------------------------------------------------------------------

current_gameweek = next(
    (e['id'] for e in events if e['is_current']),
    next((e['id'] for e in events if e['is_next']), None)
)

if current_gameweek is None:
    # Fallback: use last finished GW
    current_gameweek = max(e['id'] for e in events if e.get('finished'))

print(f"   ✅ Current Gameweek: {current_gameweek}")

next_5_gameweeks = [
    {'id': e['id'], 'name': e['name']}
    for e in events
    if e['id'] > current_gameweek
][:5]

# ---------------------------------------------------------------------------
# 3. FIXTURES
# ---------------------------------------------------------------------------

print("\n📡 Fetching fixtures...")
fixtures = fetch_fpl_data('https://fantasy.premierleague.com/api/fixtures/')
if fixtures is None:
    print("   ⚠️  No fixtures data. Continuing with empty fixtures.")
    fixtures = []

fixtures_data = fixtures   # Passed to ML model

# Build team → GW → opponent/difficulty lookup (supports double gameweeks)
team_opponents = {
    team_id: {
        gw['id']: {'opponent': [], 'difficulty': [], 'home_away': []}
        for gw in next_5_gameweeks
    }
    for team_id in teams.keys()
}

next_5_ids = {gw['id'] for gw in next_5_gameweeks}

for fixture in fixtures:
    event_id      = fixture.get('event')
    home_team_id  = fixture.get('team_h')
    away_team_id  = fixture.get('team_a')
    home_diff     = fixture.get('team_h_difficulty')
    away_diff     = fixture.get('team_a_difficulty')

    if not (event_id and home_team_id and away_team_id and event_id in next_5_ids):
        continue

    # Home team
    if home_team_id in team_opponents and event_id in team_opponents[home_team_id]:
        team_opponents[home_team_id][event_id]['opponent'].append(teams[away_team_id])
        team_opponents[home_team_id][event_id]['difficulty'].append(home_diff)
        team_opponents[home_team_id][event_id]['home_away'].append(f"{teams[home_team_id]}(H)")

    # Away team
    if away_team_id in team_opponents and event_id in team_opponents[away_team_id]:
        team_opponents[away_team_id][event_id]['opponent'].append(teams[home_team_id])
        team_opponents[away_team_id][event_id]['difficulty'].append(away_diff)
        team_opponents[away_team_id][event_id]['home_away'].append(f"{teams[away_team_id]}(A)")

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
    # retrain=False → use cached model daily (fast)
    # retrain=True  → re-fetch training data + retrain (set weekly via workflow env var)
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

    # Graceful fallback — columns must exist for sheets to write cleanly
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

# Ensure all 5 columns exist (blank if fewer than 5 GWs left in season)
for i in range(1, 6):
    if f'Next GW Difficulty {i}' not in player_df.columns:
        player_df[f'Next GW Difficulty {i}'] = np.nan
    if f'Next GW Opponent {i}' not in player_df.columns:
        player_df[f'Next GW Opponent {i}'] = ''

# Primary next GW alias
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

# Next 5 GW FDR as list (useful for Power BI custom visuals)
player_df['Next 5 GW FDR'] = player_df.apply(
    lambda row: [
        parse_difficulty(row.get(f'Next GW Difficulty {i}', np.nan))
        for i in range(1, max_next + 1)
    ],
    axis=1,
)

# ---------------------------------------------------------------------------
# 9. DASHBOARD-READY DERIVED COLUMNS
# These are needed to support the Power BI dashboard visualisations.
# ---------------------------------------------------------------------------

# Net transfers this GW (positive = being bought, negative = being sold)
player_df['Net GW Transfers'] = player_df['GW Transfers In'] - player_df['GW Transfers Out']

# Price rise probability proxy (normalised net transfer direction)
player_df['Price Rise Score'] = player_df.apply(
    lambda row: round(
        (row['GW Transfers In'] - row['GW Transfers Out']) /
        max(row['GW Transfers In'] + row['GW Transfers Out'], 1),
        3
    ),
    axis=1,
)

# Differential score: high xP + low ownership = massive rank swing potential
player_df['Differential Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) *
        (1 - float(row.get('Ownership (%)', 0) or 0) / 100) * 10,
        2
    ),
    axis=1,
)

# Captaincy score: composite of xP, form, fixture
player_df['Captaincy Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) * 0.50 +
        float(row.get('Form', 0) or 0) * 0.30 +
        (5.0 - parse_difficulty(row.get('Next GW Difficulty 1', 3))) * 0.20,
        2
    ),
    axis=1,
)

# Transfer in composite score (xP, fixture, value)
player_df['Transfer In Score'] = player_df.apply(
    lambda row: round(
        float(row.get('xP', 0) or 0) * 0.40 +
        (5.0 - parse_difficulty(row.get('Next GW Difficulty 1', 3))) * 0.40 +
        float(row.get('Points/Game', 0) or 0) * 0.20,
        2
    ),
    axis=1,
)

# Points per million (value metric)
player_df['Points Per Million'] = player_df.apply(
    lambda row: round(
        float(row.get('Total Points', 0) or 0) / max(float(row.get('Cost', 4.0) or 4.0), 1.0),
        2
    ),
    axis=1,
)

# Double/blank GW flag (basic — checks if player has 0 opponents listed)
player_df['Has DGW'] = player_df['Next GW Opponent 1'].apply(
    lambda x: '⚡ DGW' if (isinstance(x, str) and ',' in x) else ''
)
player_df['Has BGW'] = player_df['Next GW Opponent 1'].apply(
    lambda x: '⚠️ BGW' if (not isinstance(x, str) or x.strip() == '') else ''
)

# Last updated timestamp
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
        # Meta
        'Expected points Next GW', 'Current Gameweek', 'Last Updated',
    ]

    next_gw_cols = []
    for i in range(1, max_next + 1):
        next_gw_cols.append(f'Next GW Opponent {i}')
        next_gw_cols.append(f'Next GW Difficulty {i}')

    all_cols  = base_cols + next_gw_cols
    df_pos    = player_df[player_df['Position'] == position_name].copy()
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

    h_id     = fixture.get('team_h')
    a_id     = fixture.get('team_a')
    h_goals  = fixture.get('team_h_score') or 0
    a_goals  = fixture.get('team_a_score') or 0

    if not (h_id and a_id):
        continue

    teams_df.loc[teams_df['id'] == h_id, 'Goals Scored']   += h_goals
    teams_df.loc[teams_df['id'] == h_id, 'Goals Conceded'] += a_goals
    teams_df.loc[teams_df['id'] == a_id, 'Goals Scored']   += a_goals
    teams_df.loc[teams_df['id'] == a_id, 'Goals Conceded'] += h_goals
    teams_df.loc[teams_df['id'] == h_id, 'Games'] += 1
    teams_df.loc[teams_df['id'] == a_id, 'Games'] += 1

    h_short = teams_df.loc[teams_df['id'] == h_id, 'team'].values[0]
    a_short = teams_df.loc[teams_df['id'] == a_id, 'team'].values[0]
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
    ["'a'", "Available — Fully fit",                    "Start, transfer in freely"],
    ["'d'", "Doubtful — Minor injury risk",              "Check pre-deadline news"],
    ["'i'", "Injured — Not available",                  "Sell or bench immediately"],
    ["'s'", "Suspended — Banned",                       "Avoid until ban served"],
    ["'u'", "Unavailable — Non-injury reason",          "Avoid for this GW"],
    ["'n'", "Not in squad — Rotation risk",             "Sell unless price holds value"],
    [],
    ["Metric",               "Definition",                                        "FPL Usage Tip"],
    ["xP",                   "ML-predicted points next GW (ensemble model)",      "Primary pick signal — trust over Form"],
    ["xP_confidence",        "Prediction uncertainty (std dev across models)",    "Low value = high confidence. High = risky"],
    ["AI_Rating",            "Premium / Good / Average / Monitor / Avoid",        "Quick visual filter for transfers"],
    ["Captaincy Score",      "xP×0.5 + Form×0.3 + Fixture×0.2",                 "Sort descending for captain pick"],
    ["Transfer In Score",    "xP×0.4 + Fixture×0.4 + PPG×0.2",                  "Sort descending for who to buy"],
    ["Differential Score",   "xP × (1 - Ownership%) × 10",                      "High = rank-gaining differentials"],
    ["FD Index",             "Form ÷ Difficulty Score",                           "In-form players in easy fixtures"],
    ["XG",                   "Expected Goals (season)",                            "Goal threat indicator"],
    ["Delta G",              "Goals − xG",                                         "+ve = clinical finisher, −ve = may regress"],
    ["XA",                   "Expected Assists",                                   "Creative involvement proxy"],
    ["Delta GI",             "GI − (xG + xA)",                                    "+ve = outperforming, −ve = underperforming"],
    ["ΔG_GW",                "XG Current GW − XG Previous GW",                   "+ve = improving attacking momentum"],
    ["ΔGI",                  "XGI Current GW − XGI Previous GW",                 "Weekly momentum tracker"],
    ["Points Per Million",   "Total Points ÷ Cost",                               "Best value players in the price range"],
    ["Net GW Transfers",     "GW Transfers In − GW Transfers Out",               "+ve = market buying, price rise likely"],
    ["Price Rise Score",     "Normalised net transfer direction (−1 to +1)",      "+0.3+ → potential price rise"],
]

status_code_df = pd.DataFrame(status_code_info)
status_code_df['Last Updated'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M UTC')

# ---------------------------------------------------------------------------
# 13. WRITE ALL SHEETS
# ---------------------------------------------------------------------------

print("\n📝 Writing all data to Google Sheets...")

write_to_sheet(sheet, player_df,              'Player Data')
write_to_sheet(sheet, goalkeepers_smart_picks, 'Smart Picks - Goalkeepers')
write_to_sheet(sheet, defenders_smart_picks,   'Smart Picks - Defenders')
write_to_sheet(sheet, midfielders_smart_picks, 'Smart Picks - Midfielders')
write_to_sheet(sheet, forwards_smart_picks,    'Smart Picks - Forwards')
write_to_sheet(sheet, attacking_teams,         'Best Attacking Teams')
write_to_sheet(sheet, defensive_teams,         'Best Defensive Teams')
write_to_sheet(sheet, status_code_df,          'FPL Key Metrics Guide')

# ---------------------------------------------------------------------------
# 14. PIPELINE SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("✅ FPL DATA PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"  📊 Players processed   : {len(player_df)}")
print(f"  🤖 ML predictions      : {'✅ Active' if ml_success else '⚠️  Fallback (Form)'}")
print(f"  📈 New columns added   : xP · xP_confidence · AI_Rating · Captaincy Score")
print(f"                           Transfer In Score · Differential Score · FD Index")
print(f"                           Net GW Transfers · Price Rise Score · Points Per Million")
print(f"  📅 Gameweek            : GW{current_gameweek}")
print(f"  🔗 Sheet               : https://docs.google.com/spreadsheets/d/{sheet_id}")
print(f"  ⏱️  Completed           : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}")
print("="*70)
