import os
import json
import re
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import requests
import numpy as np
from google.oauth2.service_account import Credentials
import time

# ⭐ PRODUCTION MODEL IMPORT
from fpl_ml_model import add_ml_predictions

# Load service account info from env var
creds_json = os.getenv("GOOGLE_CREDENTIALS")
creds_dict = json.loads(creds_json)

# Define the required scopes
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Authenticate
credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(credentials)

# Open Google Sheet
sheet_id = os.getenv("GOOGLE_SHEET_ID")
sheet = client.open_by_key(sheet_id)

# === FUNCTION TO WRITE DATA TO SEPARATE SHEETS ===
def write_to_sheet(sheet, df, sheet_name):
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="1000", cols="30")
   
    worksheet.clear()
    set_with_dataframe(worksheet, df, include_column_header=True, resize=True)
    worksheet.freeze(rows=1, cols=2)

# ✅ HELPER FUNCTION: Safe API call
def fetch_fpl_data(url, max_retries=3):
    """Fetch data from FPL API with error handling and retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"⚠️  JSON decode error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                print(f"❌ Failed after {max_retries} attempts")
                return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                print(f"❌ Failed after {max_retries} attempts")
                return None
    return None

#1 FPL API endpoints
fpl_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

# Fetch FPL Data
print("Fetching bootstrap data...")
data = fetch_fpl_data(fpl_url)

if data is None:
    raise Exception("Failed to fetch bootstrap data from FPL API")

# Extract player data
players = data['elements']
teams = {team['id']: team['short_name'] for team in data['teams']}
positions = {position['id']: position['singular_name'] for position in data['element_types']}
events = data['events']

# Store teams data for ML model
teams_data = data['teams']

# Current Gameweek
current_gameweek = None
for event in events:
    if event['is_current']:
        current_gameweek = event['id']

# GW for next 5 gameweeks
next_5_gameweeks = [e for e in events if e['id'] > current_gameweek][:5]
next_5_gameweeks = [{'id': gw['id'], 'name': gw['name']} for gw in next_5_gameweeks]

# Fetch fixture data
fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
print("Fetching fixtures data...")
fixtures = fetch_fpl_data(fixtures_url)

if fixtures is None:
    print("⚠️  Warning: Could not fetch fixtures. Continuing with empty fixtures.")
    fixtures = []

# Store fixtures for ML model
fixtures_data = fixtures

# Create dictionary to store the opponents and difficulty scores for each team's Gameweek
team_opponents = {
    team_id: {
        gw['id']: {'opponent': [], 'difficulty': [], 'home_away': []} for gw in next_5_gameweeks
    } for team_id in teams.keys()
}

# Fill dictionary with all opponents (supports double Gameweeks)
for fixture in fixtures:
    event_id = fixture.get('event')
    home_team_id = fixture.get('team_h')
    away_team_id = fixture.get('team_a')
    home_difficulty = fixture.get('team_h_difficulty')
    away_difficulty = fixture.get('team_a_difficulty')
   
    if event_id and home_team_id and away_team_id and any(gw['id'] == event_id for gw in next_5_gameweeks):
        # Home team
        team_opponents[home_team_id][event_id]['opponent'].append(teams[away_team_id])
        team_opponents[home_team_id][event_id]['difficulty'].append(home_difficulty)
        team_opponents[home_team_id][event_id]['home_away'].append(f'{teams[home_team_id]}(H)')
       
        # Away team
        team_opponents[away_team_id][event_id]['opponent'].append(teams[home_team_id])
        team_opponents[away_team_id][event_id]['difficulty'].append(away_difficulty)
        team_opponents[away_team_id][event_id]['home_away'].append(f'{teams[away_team_id]}(A)')

# Fetch Gameweek data for expected goals
def fetch_gameweek_data(gameweek):
    url = f'https://fantasy.premierleague.com/api/event/{gameweek}/live/'
    print(f"Fetching gameweek {gameweek} data...")
    gw_data = fetch_fpl_data(url)
    return gw_data if gw_data else {}

# Fetch expected goals for last two Gameweeks
current_gw = current_gameweek
previous_gw = current_gameweek - 1 if current_gameweek and current_gameweek > 1 else None

current_gw_data = fetch_gameweek_data(current_gw) if current_gw else {}
previous_gw_data = fetch_gameweek_data(previous_gw) if previous_gw else {}

# Extract expected goals (xG) data for players
def extract_expected_goals(gw_data):
    if not gw_data or 'elements' not in gw_data:
        return {}
    players_stats = gw_data['elements']
    xg_data = {}
    for player in players_stats:
        player_id = player['id']
        stats = player['stats']
        xg_data[player_id] = stats.get('expected_goals', 0)
    return xg_data

def extract_expected_goal_involvements(gw_data):
    if not gw_data or 'elements' not in gw_data:
        return {}
    players_stats = gw_data['elements']
    xgi_data = {}
    for player in players_stats:
        player_id = player['id']
        stats = player['stats']
        xgi_data[player_id] = stats.get('expected_goal_involvements', 0)
    return xgi_data

current_gw_xg = extract_expected_goals(current_gw_data)
previous_gw_xg = extract_expected_goals(previous_gw_data)
current_gw_xgi = extract_expected_goal_involvements(current_gw_data)
previous_gw_xgi = extract_expected_goal_involvements(previous_gw_data)

# Create a list of player information
player_info = []
for player in players:
    player_data = {
        'Photo': player['photo'],
        'Player ID': player['id'],
        'Player Name': player['web_name'],
        'First Name': player['first_name'],
        'Last Name': player['second_name'],
        'Form': player['form'],
        'Team': teams[player['team']],
        'Position': positions[player['element_type']],
        'Defensive Contributions': player['defensive_contribution'],
        'Defensive Contributions/90': player['defensive_contribution_per_90'],
        'Cost': player['now_cost'] / 10,
        'GW Points': player['event_points'],
        'Expected points Current GW': player['ep_this'],
        'Expected points Next GW': player['ep_next'],
        'Total Points': player['total_points'],
        'Points/Game': player['points_per_game'],
        'Goals': player['goals_scored'],
        'Assists': player['assists'],
        'Clean Sheets': player['clean_sheets'],
        'Saves': player['saves'],
        'Minutes': player['minutes'],
        'Ownership (%)': player['selected_by_percent'],
        'Transfers In': player['transfers_in'],
        'GW Transfers In': player['transfers_in_event'],
        'Transfers Out': player['transfers_out'],
        'GW Transfers Out': player['transfers_out_event'],
        'Form Rank': player['form_rank'],
        'Form Rank/Position': player['form_rank_type'],
        'Points/Game Rank': player['points_per_game_rank'],
        'Position Ranking': player['points_per_game_rank_type'],
        'Yellow Cards': player['yellow_cards'],
        'Red Cards': player['red_cards'],
        'Penalties Saved': player['penalties_saved'],
        'Penalties Missed': player['penalties_missed'],
        'Penalty Order': player['penalties_order'],
        'Freekick/Cornerkick Order': player['corners_and_indirect_freekicks_order'],
        'Chance of playing next': player['chance_of_playing_next_round'],
        'Cost Change/GW': player['cost_change_event'],
        'In Dream Team': player['in_dreamteam'],
        'DreamTeam Count': player['dreamteam_count'],
        'Influence': player['influence'],
        'Creativity': player['creativity'],
        'Threat': player['threat'],
        'ICT Index': player['ict_index'],
        'ICT Index Rank': player['ict_index_rank'],
        'Goals Conceded': player['goals_conceded'],
        'Total Bonus Point': player['bonus'],
        'BPS': player['bps'],
        'Availability': player['status'],
        'Starts': player['starts'],
        'Starts/90': player['starts_per_90'],
        'Clean Sheets/90': player['clean_sheets_per_90'],
        'Saves/90': player['saves_per_90'],
        'XG': player['expected_goals'],
        'XA': player['expected_assists'],
        'XG/90': player['expected_goals_per_90'],
        'XA/90': player['expected_assists_per_90'],
        'XGI': player['expected_goal_involvements'],
        'XGC': player['expected_goals_conceded'],
        'XGI/90': player['expected_goal_involvements_per_90'],
        'XGC/90': player['expected_goals_conceded_per_90'],
        'Goals Conceded/90': player['goals_conceded_per_90'],
        'Current Gameweek': current_gameweek
    }

    # Add next 5 Gameweeks' opponents and difficulty scores with actual GW names
    for i, gw in enumerate(next_5_gameweeks, start=1):
        opponent_info = team_opponents[player['team']][gw['id']]

        opponents_with_venue = []
        for opp, venue in zip(opponent_info['opponent'], opponent_info['home_away']):
            opponents_with_venue.append(f"{opp}{venue[-3:]}")

        opponent_combined = ', '.join(opponents_with_venue)
        difficulty_combined = ', '.join(map(str, opponent_info['difficulty']))

        player_data[f'{gw["name"]}'] = opponent_combined
        player_data[f'{gw["name"]} Difficulty'] = difficulty_combined

    player_data['XG Current GW'] = float(current_gw_xg.get(player['id'], 0))
    player_data['XG Previous GW'] = float(previous_gw_xg.get(player['id'], 0))
    player_data['ΔG_GW'] = player_data['XG Current GW'] - player_data['XG Previous GW']  

    player_data['XGI Current GW'] = float(current_gw_xgi.get(player['id'], 0))  
    player_data['XGI Previous GW'] = float(previous_gw_xgi.get(player['id'], 0))
    player_data['ΔGI'] = player_data['XGI Current GW'] - player_data['XGI Previous GW']

    player_data['Goals'] = float(player_data['Goals'])
    player_data['Assists'] = float(player_data['Assists'])
    player_data['XG'] = float(player_data['XG'])
    player_data['XGI'] = float(player_data['XGI'])

    player_data['GI'] = player_data['Goals'] + player_data['Assists']
    player_data['Delta G'] = player_data['Goals'] - player_data['XG']
    player_data['Delta GI'] = player_data['GI'] - player_data['XGI']

    player_info.append(player_data)

# Convert the list of player information into a DataFrame
player_df = pd.DataFrame(player_info)


# ⭐⭐⭐ PRODUCTION ML MODEL  ⭐⭐⭐
# =============================================================================
# PRODUCTION ML PREDICTION - Position-Specific Ensemble Models
# =============================================================================
try:
    print("\n" + "="*70)
    print("🤖 PRODUCTION MACHINE LEARNING PREDICTIONS")
    print("="*70)
    
    # Set retrain=True to retrain models (weekly)
    # Set retrain=False to use cached models (daily)
    retrain_model = False
    
    player_df, ml_model = add_ml_predictions_v2(
        player_df, 
        teams_data, 
        fixtures_data, 
        retrain=retrain_model
    )
    
    print(f"✅ Production ML predictions added to player_df")
    print(f"   New columns: xP, xP_confidence, AI_Rating")
    
except Exception as e:
    print(f"⚠️  ML prediction failed: {str(e)}")
    print("   Continuing with standard metrics...")
    # Add dummy columns so code doesn't break
    player_df['xP'] = player_df['Form'].astype(float)
    player_df['xP_confidence'] = 0
    player_df['AI_Rating'] = 'N/A'


# Normalize dynamic Gameweek columns
difficulty_cols = [col for col in player_df.columns if re.match(r'Gameweek\s+\d+\s+Difficulty$', col)]
opponent_cols = [col for col in player_df.columns if re.match(r'Gameweek\s+\d+$', col)]

def gw_number(colname):
    m = re.search(r'Gameweek\s+(\d+)', colname)
    return int(m.group(1)) if m else 0

difficulty_cols_sorted = sorted(difficulty_cols, key=gw_number)
opponent_cols_sorted = sorted(opponent_cols, key=gw_number)

for i, col in enumerate(difficulty_cols_sorted, start=1):
    new_name = f'Next GW Difficulty {i}'
    player_df.rename(columns={col: new_name}, inplace=True)

for i, col in enumerate(opponent_cols_sorted, start=1):
    new_name = f'Next GW Opponent {i}'
    player_df.rename(columns={col: new_name}, inplace=True)

max_next = max(1, len(next_5_gameweeks))
for i in range(1, max_next + 1):
    diff_col = f'Next GW Difficulty {i}'
    opp_col = f'Next GW Opponent {i}'
    if diff_col not in player_df.columns:
        player_df[diff_col] = np.nan
    if opp_col not in player_df.columns:
        player_df[opp_col] = ''

player_df['Next GW Difficulty'] = player_df['Next GW Difficulty 1']
player_df['Next 5 GW FDR'] = player_df.apply(
    lambda row: [row.get(f'Next GW Difficulty {i}', np.nan) for i in range(1, max_next + 1)],
    axis=1
)
player_df['Next GW Opponent'] = player_df['Next GW Opponent 1']

pd.set_option('display.max.columns', 75)

# Helper function to safely parse Difficulty values
def parse_difficulty(val):
    if isinstance(val, list):
        return sum(val)
    elif isinstance(val, str):
        try:
            return sum(float(x.strip()) for x in val.split(',') if x.strip() != '')
        except ValueError:
            return 0
    elif isinstance(val, (int, float)):
        return float(val)
    return 0

# Calculate the summation of the difficulty scores for the next N gameweeks
player_df['Difficulty Score'] = player_df.apply(
    lambda row: sum(
        parse_difficulty(row.get(f'Next GW Difficulty {i}', 0))
        for i in range(1, max_next + 1)
    ), axis=1
)

# FD Index calculation
player_df['FD Index'] = player_df.apply(
    lambda r: round(float(r['Form']) / r['Difficulty Score'], 2) if r['Difficulty Score'] not in (0, np.nan) else np.nan,
    axis=1
)

# Next GW Difficulty: parse the first upcoming GW difficulty
player_df['Next GW Difficulty'] = player_df['Next GW Difficulty'].apply(parse_difficulty)

# Ensure Next 5 GW FDR list is numeric
player_df['Next 5 GW FDR'] = player_df['Next 5 GW FDR'].apply(
    lambda lst: [parse_difficulty(x) for x in lst]
)

for i in range(1, max_next + 1):
    player_df[f'Next GW Opponent {i}'] = player_df.get(f'Next GW Opponent {i}', '')

# ⭐ Table creation function with xP columns
def create_Gw_transfers_in_table(position_name):
    base_cols = [
        'Player Name','Availability', 'Team', 'Position', 'Cost', 
        'xP', 'xP_confidence', 'AI_Rating',  # ML COLUMNS
        'Form', 'FD Index', 'XG', 'Clean Sheets', 'Saves', 'Starts', 'Minutes', 
        'Yellow Cards', 'Red Cards', 'Defensive Contributions', 'Defensive Contributions/90', 
        'Goals', 'Assists', 'XG Current GW','XG Previous GW', 'ΔG_GW', 'Delta G', 
        'XA', 'Delta GI', 'XG/90', 'Ownership (%)', 'GW Points', 'Points/Game',
        'Expected points Next GW', 'Total Points', 'Difficulty Score', 
        'Total Bonus Point', 'Current Gameweek'
    ]
    
    next_gw_cols = []
    for i in range(1, max_next + 1):
        next_gw_cols.append(f'Next GW Opponent {i}')
        next_gw_cols.append(f'Next GW Difficulty {i}')
    
    final_cols = base_cols + next_gw_cols + ['GW Transfers In', 'GW Transfers Out']
    
    df = player_df[player_df['Position'] == position_name]
    selected_cols = [c for c in final_cols if c in df.columns]
    return df[selected_cols].sort_values(by='GW Transfers In', ascending=False)


# Create transfer pick tables for each position
goalkeepers_Gw_transfers_in = create_Gw_transfers_in_table('Goalkeeper')
defenders_Gw_transfers_in = create_Gw_transfers_in_table('Defender')
midfielders_Gw_transfers_in = create_Gw_transfers_in_table('Midfielder')
forwards_Gw_transfers_in = create_Gw_transfers_in_table('Forward')

   
#3 === FETCH DATA AND PREPARE TEAM DATA ===
teams_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'

print("\nFetching team stats data...")
teams_bootstrap = fetch_fpl_data(teams_url)
fixtures_data_full = fetch_fpl_data(fixtures_url)

if teams_bootstrap is None or fixtures_data_full is None:
    print("⚠️  Warning: Could not fetch team/fixtures data. Using minimal team info.")
    teams_data_df = data['teams']
    fixtures_data_full = []
else:
    teams_data_df = teams_bootstrap['teams']

# Process team data into DataFrame
teams_df = pd.DataFrame(teams_data_df)
teams_df = teams_df[['id', 'short_name', 'name', 'played', 'points', 'form']]
teams_df.rename(columns={'short_name': 'team', 'name': 'Team'}, inplace=True)

# Initialize columns for goals and games
teams_df['Goals Scored'] = 0
teams_df['Goals Conceded'] = 0
teams_df['Games'] = 0
teams_df['Last 5 GW Results'] = ""

# Initialize result tracking per team as a dictionary
team_results = {team_id: [] for team_id in teams_df['id']}

# Process fixtures to gather results and calculate goals
for fixture in fixtures_data_full:
    if fixture.get('finished') is False:
        continue

    home_team_id = fixture.get('team_h')
    away_team_id = fixture.get('team_a')
    home_goals = fixture.get('team_h_score')
    away_goals = fixture.get('team_a_score')

    if home_team_id is None or away_team_id is None:
        continue

    home_goals = home_goals if home_goals is not None else 0
    away_goals = away_goals if away_goals is not None else 0

    teams_df.loc[teams_df['id'] == home_team_id, 'Goals Scored'] += home_goals
    teams_df.loc[teams_df['id'] == home_team_id, 'Goals Conceded'] += away_goals
    teams_df.loc[teams_df['id'] == away_team_id, 'Goals Scored'] += away_goals
    teams_df.loc[teams_df['id'] == away_team_id, 'Goals Conceded'] += home_goals

    teams_df.loc[teams_df['id'] == home_team_id, 'Games'] += 1
    teams_df.loc[teams_df['id'] == away_team_id, 'Games'] += 1

    home_team_name = teams_df.loc[teams_df['id'] == home_team_id, 'team'].values[0]
    away_team_name = teams_df.loc[teams_df['id'] == away_team_id, 'team'].values[0]

    result = f"{home_team_name} {home_goals} - {away_goals} {away_team_name}"

    team_results[home_team_id].append(result)
    team_results[away_team_id].append(result)

teams_df['Last 5 GW Results'] = teams_df['id'].apply(lambda tid: ', '.join(team_results[tid][-5:]))

teams_df['Goals Scored/Game'] = (
    teams_df['Goals Scored'] / teams_df['Games'].replace(0, np.nan)
).round(1)

teams_df['Goals Conceded/Game'] = (
    teams_df['Goals Conceded'] / teams_df['Games'].replace(0, np.nan)
).round(1)

teams_df['Goal Difference'] = teams_df['Goals Scored'] - teams_df['Goals Conceded']

# Create the tables
attacking_teams = teams_df.sort_values(by=['Goals Scored/Game', 'Goals Scored', 'Goal Difference'], ascending=[False, False, False])[[
    'Team', 'Games', 'Goals Scored/Game', 'Goals Conceded/Game', 'Goals Scored',
    'Goals Conceded', 'Goal Difference', 'Last 5 GW Results'
]]
attacking_teams['Last Updated'] = pd.to_datetime('now')

defensive_teams = teams_df.sort_values(by=['Goals Conceded/Game', 'Goals Conceded'], ascending=[True, True])[[  
    'Team', 'Games', 'Goals Conceded/Game', 'Goals Conceded', 'Goals Scored/Game',
    'Goals Scored', 'Goal Difference', 'Last 5 GW Results'
]]
defensive_teams['Last Updated'] = pd.to_datetime('now')

# Create the status code info
status_code_info = [
    ["Status Code", "Meaning / Implication for FPL"],
    ["'a'", "Available – Fit to play, no restrictions"],
    ["'d'", "Doubtful – Minor injury or illness risk"],
    ["'i'", "Injured – Not available to play"],
    ["'s'", "Suspended – Missing due to red/yellow cards"],
    ["'u'", "Unavailable – Non-injury reason (e.g. transfer, international duty)"],
    ["'n'", "Not in squad – Possibly dropped or rotated"],
   
    [],
    ["Metric", "Definition", "FPL Usage Tip"],
    ["FD Index (Form / Fixture Difficulty)",
     "Highlights in-form players with favorable upcoming fixtures",
     "Pick players in form with easy matches ahead"],
   
    ["xG (Expected Goals)",
     "Predicts goal-scoring potential based on shot quality",
     "Helps find players likely to score"],
   
    ["Delta G (Goals - xG)",
     "Reveals finishing efficiency or over/underperformance",
     "Spot clinical finishers or potential regressions"],
   
    ["xA (Expected Assists)",
     "Gauges assist potential from creative passing",
     "Pick players with high assist potential"],
   
    ["ΔG_gw (Delta G/gameweek = Current GW XG - Previous GW XG)",
     "How much a player's involvement in expected goals has changed from last week to this week",
     "Track weekly attacking momentum (+ve = in-form, -ve = declining)"],
   
    ["Delta GI (Goal Involvements - [xG + xA])",
     "Shows actual impact vs expected contribution",
     "Identify players outperforming stats (+ve = exceeding expectations)"],
    
    ["xP (Expected Points - ML Prediction)",
     "AI prediction of next gameweek points using 20 GW history",
     "Production model with position-specific algorithms and opponent analysis"],
    
    ["xP_confidence",
     "Uncertainty range for xP prediction",
     "Lower = more confident. High variance players have higher values"],
    
    ["AI_Rating",
     "Premium/Good/Medium/Avoid based on xP",
     "Quick visual indicator for transfer decisions"]
]

status_code_df = pd.DataFrame(status_code_info)
status_code_df['Last Updated'] = pd.to_datetime('now')

# === WRITE ALL DATA TO SEPARATE SHEETS ===
print("\nWriting data to Google Sheets...")
write_to_sheet(sheet, player_df, 'Player Data')
write_to_sheet(sheet, goalkeepers_Gw_transfers_in.head(5), 'Smart Picks - Goalkeepers')
write_to_sheet(sheet, defenders_Gw_transfers_in.head(10), 'Smart Picks - Defenders')
write_to_sheet(sheet, midfielders_Gw_transfers_in.head(15), 'Smart Picks - Midfielders')
write_to_sheet(sheet, forwards_Gw_transfers_in.head(5), 'Smart Picks - Forwards')
write_to_sheet(sheet, attacking_teams, 'Best Attacking Teams')
write_to_sheet(sheet, defensive_teams, 'Best Defensive Teams')
write_to_sheet(sheet, status_code_df, 'FPL Key Metrics Guide')

print("\n" + "="*70)
print("✅ FPL DATA PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"📊 Total players processed: {len(player_df)}")
print(f"🤖 ML predictions: {'Active (Production Model)' if 'xP' in player_df.columns else 'Failed (using fallback)'}")
print(f"📈 Data uploaded to: https://docs.google.com/spreadsheets/d/{sheet_id}")
print("="*70)
