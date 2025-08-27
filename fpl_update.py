import os
import json
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import requests
import numpy as np

# === HANDLE CREDS FROM GITHUB ACTIONS ===
creds_json = os.getenv("GOOGLE_CREDENTIALS")
if creds_json:
    with open("creds.json", "w") as f:
        f.write(creds_json)

# === AUTH SETUP ===
sheet_id = os.getenv("GOOGLE_SHEET_ID")
creds_path = "creds.json"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
client = gspread.authorize(creds)

sheet = client.open_by_key(sheet_id)


# === FUNCTION TO WRITE DATA TO SEPARATE SHEETS ===
def write_to_sheet(sheet, df, sheet_name):
    # Create or open the sheet
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="1000", cols="20")
    
    worksheet.clear()  # Clear previous content
    set_with_dataframe(worksheet, df, include_column_header=True, resize=True)
    worksheet.freeze(rows=1, cols=2)

#1 FPL API endpoints
fpl_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

# Fetch FPL Data
response = requests.get(fpl_url)
data = response.json()

# Extract player data
players = data['elements']
teams = {team['id']: team['short_name'] for team in data['teams']}  # Use short names for teams
positions = {position['id']: position['singular_name'] for position in data['element_types']}
events = data['events']

# Current Gameweek and the next 5 Gameweeks
current_gameweek = None
next_5_gameweeks = []
found_next = False

for event in events:
    if event['is_current']:
        current_gameweek = event['id']
    if event['is_next']:
        found_next = True
    if found_next and len(next_5_gameweeks) < 5:
        next_5_gameweeks.append({'id': event['id'], 'name': event['name']})

# Fetch fixture data
fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
fixtures_response = requests.get(fixtures_url)

# Check for valid JSON
try:
    fixtures = fixtures_response.json()
except requests.exceptions.JSONDecodeError:
    print("Error: Received an invalid JSON response from the fixtures endpoint.")
    print("Response content:", fixtures_response.text)
    fixtures = []

# Create dictionary to store the opponents and difficulty scores for each team's Gameweek
team_opponents = {
    team_id: {
        gw['id']: {'opponent': [], 'difficulty': [], 'home_away': []} for gw in next_5_gameweeks
    } for team_id in teams.keys()
}

# Fill dictionary with all opponents (supports double Gameweeks)
for fixture in fixtures:
    event_id = fixture['event']
    home_team_id = fixture['team_h']
    away_team_id = fixture['team_a']
    home_difficulty = fixture['team_h_difficulty']
    away_difficulty = fixture['team_a_difficulty']
    
    if any(gw['id'] == event_id for gw in next_5_gameweeks):
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
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data for Gameweek {gameweek}")

# Fetch expected goals for last two Gameweeks
current_gw = current_gameweek  # Current GW
previous_gw = current_gameweek - 1 if current_gameweek and current_gameweek > 1 else None  # Only valid if GW > 1

current_gw_data = fetch_gameweek_data(current_gw) if current_gw else {}
previous_gw_data = fetch_gameweek_data(previous_gw) if previous_gw else {}

# Extract expected goals (xG) data for players
def extract_expected_goals(gw_data):
    players_stats = gw_data['elements']
    xg_data = {}
    for player in players_stats:
        player_id = player['id']
        stats = player['stats']
        xg_data[player_id] = stats['expected_goals']
    return xg_data

def extract_expected_goal_involvements(gw_data):
    players_stats = gw_data['elements']
    xgi_data = {}
    for player in players_stats:
        player_id = player['id']
        stats = player['stats']
        xgi_data[player_id] = stats['expected_goal_involvements']
    return xgi_data

# Extract xG and xGI only if the data exists
current_gw_xg = extract_expected_goals(current_gw_data) if current_gw_data else {}
previous_gw_xg = extract_expected_goals(previous_gw_data) if previous_gw_data else {}

current_gw_xgi = extract_expected_goal_involvements(current_gw_data) if current_gw_data else {}
previous_gw_xgi = extract_expected_goal_involvements(previous_gw_data) if previous_gw_data else {}

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
        'Cost': player['now_cost'] / 10,  # Cost is in tenths of a million
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
    # Add next 5 Gameweeks' opponents and difficulty scores with the gameweek names
    for i, gw in enumerate(next_5_gameweeks, start=1):
        opponent_info = team_opponents[player['team']][gw['id']]

        # Combine multiple opponents if double GW
        opponents_with_venue = []
        for opp, venue in zip(opponent_info['opponent'], opponent_info['home_away']):
            opponents_with_venue.append(f"{opp}{venue[-3:]}")

        # Join opponents (e.g., "ARS(H), MCI(A)" if double)
        opponent_combined = ', '.join(opponents_with_venue)

        # Join difficulties (e.g., [3, 5] becomes "3, 5")
        difficulty_combined = ', '.join(map(str, opponent_info['difficulty']))

        player_data[f'{gw["name"]}'] = opponent_combined
        player_data[f'{gw["name"]} Difficulty'] = difficulty_combined
        
        # Add expected goals for last two Gameweeks to calculate change in delta G & change in delta GI
    player_data['XG Current GW'] = float(current_gw_xg.get(player['id'], 0))
    player_data['XG Previous GW'] = float(previous_gw_xg.get(player['id'], 0))
    player_data['ΔG_GW'] = player_data['XG Current GW'] - player_data['XG Previous GW']  
    
    player_data['XGI Current GW'] = float(current_gw_xgi.get(player['id'], 0))  
    player_data['XGI Previous GW'] = float(previous_gw_xgi.get(player['id'], 0))
    player_data['ΔGI'] = player_data['XGI Current GW'] - player_data['XGI Previous GW'] 

    # Convert columns to float where necessary
    player_data['Goals'] = float(player_data['Goals'])
    player_data['Assists'] = float(player_data['Assists'])
    player_data['XG'] = float(player_data['XG'])
    player_data['XGI'] = float(player_data['XGI'])

    # Calculate Goal Involvements (GI) as Goals + Assists
    player_data['GI'] = player_data['Goals'] + player_data['Assists']

    # Calculate Delta G (Goals - Expected Goals)
    player_data['Delta G'] = player_data['Goals'] - player_data['XG']

    # Calculate Delta GI (Goal Involvements - Expected Goal Involvements)
    player_data['Delta GI'] = player_data['GI'] - player_data['XGI']

    player_info.append(player_data)

# Convert the list of player information into a DataFrame
player_df = pd.DataFrame(player_info)


pd.set_option('display.max.columns', 75)
#2 Helper function to safely parse Difficulty values
def parse_difficulty(val):
    if isinstance(val, list):
        return sum(val)
    elif isinstance(val, str):
        # Split by comma and convert to float
        try:
            return sum(float(x.strip()) for x in val.split(','))
        except ValueError:
            return 0  # Handle malformed values gracefully
    elif isinstance(val, (int, float)):
        return float(val)
    return 0

# Calculate the summation of the difficulty scores for the next 5 gameweeks
player_df['Difficulty Score'] = player_df.apply(
    lambda row: sum(
        parse_difficulty(row.get(f'{gw["name"]} Difficulty', 0))
        for gw in next_5_gameweeks
    ), axis=1
)

# FD Index calculation (for the next 5 gameweeks)
player_df['FD Index'] = (player_df['Form'].astype(float) / player_df['Difficulty Score']).round(2)

# Next GW Difficulty: Sum of difficulties in first upcoming GW (handles double fixtures)
player_df['Next GW Difficulty'] = player_df[f'{next_5_gameweeks[0]["name"]} Difficulty'].apply(parse_difficulty)

# Add a column that includes the difficulty scores for the next 5 gameweeks as a list
player_df['Next 5 GW FDR'] = player_df.apply(
    lambda row: [
        parse_difficulty(row.get(f'{gw["name"]} Difficulty')) for gw in next_5_gameweeks
    ], axis=1
)

# Add columns for the next 5 opponents using the gameweek names
for gw in next_5_gameweeks:
    player_df[f'{gw["name"]} Opponent'] = player_df.apply(
        lambda row: row.get(f'{gw["name"]}', ''), axis=1
    )

# Table creation function
def create_Gw_transfers_in_table(position_name):
    return player_df[player_df['Position'] == position_name][[
        'Player Name','Availability', 'Team', 'Cost', 'Form', 'FD Index', 'XG', 'Clean Sheets', 'Goals', 'Assists', 'XG Current GW','XG Previous GW', 'ΔG_GW', 'Delta G', 'XA', 'Delta GI', 'XG/90', 'Ownership (%)', 'GW Points', 'Expected points Next GW', 'Total Points', 'Difficulty Score', 
    ] + [col for gw in next_5_gameweeks for col in [f'{gw["name"]}', f'{gw["name"]} Difficulty']] + [
        'GW Transfers In', 'GW Transfers Out'
    ]].sort_values(by='GW Transfers In', ascending=False)

# Create transfer pick tables for each position
goalkeepers_Gw_transfers_in = create_Gw_transfers_in_table('Goalkeeper')
defenders_Gw_transfers_in = create_Gw_transfers_in_table('Defender')
midfielders_Gw_transfers_in = create_Gw_transfers_in_table('Midfielder')
forwards_Gw_transfers_in = create_Gw_transfers_in_table('Forward')
#managers_Gw_transfers_in = create_Gw_transfers_table('Manager')
    
    
#3 === FETCH DATA AND PREPARE TEAM DATA ===
# Fetch data from FPL API
teams_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'

teams_data = requests.get(teams_url).json()['teams']
fixtures_data = requests.get(fixtures_url).json()

# Process team data into DataFrame with short names
teams_df = pd.DataFrame(teams_data)
teams_df = teams_df[['id', 'short_name', 'name', 'played', 'points', 'form']]
teams_df.rename(columns={'short_name': 'team'}, inplace=True)
teams_df.rename(columns={'name': 'Team'}, inplace=True)

# Initialize columns for goals and games
teams_df['Goals Scored'] = 0
teams_df['Goals Conceded'] = 0
teams_df['Games'] = 0
teams_df['Last 5 GW Results'] = ""

# Initialize result tracking per team as a dictionary
team_results = {team_id: [] for team_id in teams_df['id']}

# Process fixtures to gather results and calculate goals
for fixture in fixtures_data:
    if fixture['finished'] is False:  # Skip unfinished fixtures
        continue

    home_team_id = fixture['team_h']
    away_team_id = fixture['team_a']

    home_goals = fixture['team_h_score'] if fixture['team_h_score'] is not None else 0
    away_goals = fixture['team_a_score'] if fixture['team_a_score'] is not None else 0

    # Increment goals and games played
    teams_df.loc[teams_df['id'] == home_team_id, 'Goals Scored'] += home_goals
    teams_df.loc[teams_df['id'] == home_team_id, 'Goals Conceded'] += away_goals
    teams_df.loc[teams_df['id'] == away_team_id, 'Goals Scored'] += away_goals
    teams_df.loc[teams_df['id'] == away_team_id, 'Goals Conceded'] += home_goals

    teams_df.loc[teams_df['id'] == home_team_id, 'Games'] += 1
    teams_df.loc[teams_df['id'] == away_team_id, 'Games'] += 1

    # Capture result string
    home_team_name = teams_df.loc[teams_df['id'] == home_team_id, 'team'].values[0]
    away_team_name = teams_df.loc[teams_df['id'] == away_team_id, 'team'].values[0]

    result = f"{home_team_name} {home_goals} - {away_goals} {away_team_name}"

    # Append result to each team's history
    team_results[home_team_id].append(result)
    team_results[away_team_id].append(result)

# Assign only the last 5 results to the teams_df
teams_df['Last 5 GW Results'] = teams_df['id'].apply(lambda tid: ', '.join(team_results[tid][-5:]))

# Calculate additional metrics
#teams_df['Goals Scored/Game'] = (teams_df['Goals Scored'] / teams_df['Games'].replace(0, pd.NA)).round(1)
#teams_df['Goals Conceded/Game'] = (teams_df['Goals Conceded'] / teams_df['Games'].replace(0, pd.NA)).round(1)
#teams_df['Goal Difference'] = teams_df['Goals Scored'] - teams_df['Goals Conceded']

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
]].rename(columns={
    'team': 'Team'
})
attacking_teams['Last Updated'] = pd.to_datetime('now')
defensive_teams = teams_df.sort_values(by=['Goals Conceded/Game', 'Goals Conceded'], ascending=[True, True])[[  
    'Team', 'Games', 'Goals Conceded/Game', 'Goals Conceded', 'Goals Scored/Game', 
    'Goals Scored', 'Goal Difference', 'Last 5 GW Results'
]].rename(columns={
    'team': 'Team'
})
defensive_teams['Last Updated'] = pd.to_datetime('now')

# Create the status code message as a list of lists (each sublist is a row)
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
    
    ["ΔG_gw (Also Delta G/gameweek = Current GW XG - Previous GW XG)",
     "How much a player’s involvement in expected goals has changed from last week to this week",
     "Helps you track a player’s weekly attacking momentum(+ve ΔG_gw means this week's in-form players while -ve ΔG_gw are perfoming less than the previous week)"],
    
    ["Delta GI (Goal Involvements - [xG + xA])", 
     "Shows actual impact vs expected contribution", 
     "Identify players who consistently outperform stats(positive XG are performing more than expected while negative XG are performing below expectation)"]
]
# Convert the list of lists to a DataFrame
status_code_df = pd.DataFrame(status_code_info)

status_code_df['Last Updated'] = pd.to_datetime('now')
# === PREPARE SPLIT DATAFRAMES FOR PLAYER DATA ===

# === WRITE ALL DATA TO SEPARATE SHEETS ===
# Write Player Data table to its sheet
write_to_sheet(sheet, player_df, 'Player Data')

# Write Transfer Picks (for each position) to their respective sheets
write_to_sheet(sheet, goalkeepers_Gw_transfers_in.head(5), 'Smart Picks - Goalkeepers')
write_to_sheet(sheet, defenders_Gw_transfers_in.head(10), 'Smart Picks - Defenders')
write_to_sheet(sheet, midfielders_Gw_transfers_in.head(15), 'Smart Picks - Midfielders')
write_to_sheet(sheet, forwards_Gw_transfers_in.head(5), 'Smart Picks - Forwards')
#write_to_sheet(sheet, managers_Gw_transfers_in.head(20), 'Smart Picks - Managers')
# Write Best Attacking and Defensive Teams to their respective sheets
write_to_sheet(sheet, attacking_teams, 'Best Attacking Teams')
write_to_sheet(sheet, defensive_teams, 'Best Defensive Teams')

write_to_sheet(sheet, status_code_df, 'FPL Key Metrics Guide')
