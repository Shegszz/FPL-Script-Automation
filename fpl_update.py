import os
import json
import re
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
import requests

# ================= AUTHENTICATION =================
creds_json = os.getenv("GOOGLE_CREDENTIALS")
creds_dict = json.loads(creds_json)

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(credentials)

sheet_id = os.getenv("GOOGLE_SHEET_ID")
sheet = client.open_by_key(sheet_id)

# ================= HELPER FUNCTIONS =================
def write_to_sheet(sheet, df, sheet_name):
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="1000", cols="50")
    worksheet.clear()
    set_with_dataframe(worksheet, df, include_column_header=True, resize=True)
    worksheet.freeze(rows=1, cols=2)

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

# ================= FPL DATA FETCH =================
fpl_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'

data = requests.get(fpl_url).json()
fixtures = requests.get(fixtures_url).json()

players = data['elements']
teams = {team['id']: team['short_name'] for team in data['teams']}
positions = {position['id']: position['singular_name'] for position in data['element_types']}
events = data['events']

# Current GW and next 5 GWs
current_gameweek = next((e['id'] for e in events if e['is_current']), None)
next_5_gameweeks = [e for e in events if e['id'] > current_gameweek][:5] if current_gameweek else []

# ================= TEAM OPPONENTS & DIFFICULTY =================
team_opponents = {
    team_id: {gw['id']: {'opponent': [], 'difficulty': [], 'home_away': []} for gw in next_5_gameweeks} 
    for team_id in teams.keys()
}

for fixture in fixtures:
    gw_id = fixture['event']
    if gw_id not in [gw['id'] for gw in next_5_gameweeks]:
        continue
    home, away = fixture['team_h'], fixture['team_a']
    home_diff, away_diff = fixture['team_h_difficulty'], fixture['team_a_difficulty']

    team_opponents[home][gw_id]['opponent'].append(teams[away])
    team_opponents[home][gw_id]['difficulty'].append(home_diff)
    team_opponents[home][gw_id]['home_away'].append(f'{teams[home]}(H)')

    team_opponents[away][gw_id]['opponent'].append(teams[home])
    team_opponents[away][gw_id]['difficulty'].append(away_diff)
    team_opponents[away][gw_id]['home_away'].append(f'{teams[away]}(A)')

# ================= FETCH GAMEWEEK DATA FOR xG & xGI =================
def fetch_gameweek_data(gw):
    url = f'https://fantasy.premierleague.com/api/event/{gw}/live/'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

def extract_expected_goals(gw_data):
    xg_data = {}
    for player in gw_data.get('elements', []):
        xg_data[player['id']] = player['stats']['expected_goals']
    return xg_data

def extract_expected_goal_involvements(gw_data):
    xgi_data = {}
    for player in gw_data.get('elements', []):
        xgi_data[player['id']] = player['stats']['expected_goal_involvements']
    return xgi_data

previous_gw = current_gameweek - 1 if current_gameweek and current_gameweek > 1 else None
current_gw_data = fetch_gameweek_data(current_gameweek) if current_gameweek else {}
previous_gw_data = fetch_gameweek_data(previous_gw) if previous_gw else {}

current_gw_xg = extract_expected_goals(current_gw_data)
previous_gw_xg = extract_expected_goals(previous_gw_data)
current_gw_xgi = extract_expected_goal_involvements(current_gw_data)
previous_gw_xgi = extract_expected_goal_involvements(previous_gw_data)

# ================= PLAYER DATAFRAME =================
player_info = []
for player in players:
    pdata = {
        'Photo': player['photo'],
        'Player ID': player['id'],
        'Player Name': player['web_name'],
        'First Name': player['first_name'],
        'Last Name': player['second_name'],
        'Form': player['form'],
        'Team': teams[player['team']],
        'Position': positions[player['element_type']],
        'Cost': player['now_cost']/10,
        'GW Points': player['event_points'],
        'Expected points Current GW': player['ep_this'],
        'Expected points Next GW': player['ep_next'],
        'Total Points': player['total_points'],
        'Points/Game': player['points_per_game'],
        'Goals': float(player['goals_scored']),
        'Assists': float(player['assists']),
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
        'XG': float(player['expected_goals']),
        'XA': float(player['expected_assists']),
        'XG/90': player['expected_goals_per_90'],
        'XA/90': player['expected_assists_per_90'],
        'XGI': float(player['expected_goal_involvements']),
        'XGC': player['expected_goals_conceded'],
        'XGI/90': player['expected_goal_involvements_per_90'],
        'XGC/90': player['expected_goals_conceded_per_90'],
        'Goals Conceded/90': player['goals_conceded_per_90'],
        'Current Gameweek': current_gameweek,
        'XG Current GW': float(current_gw_xg.get(player['id'], 0)),
        'XG Previous GW': float(previous_gw_xg.get(player['id'], 0)),
        'ΔG_GW': float(current_gw_xg.get(player['id'], 0)) - float(previous_gw_xg.get(player['id'], 0)),
        'XGI Current GW': float(current_gw_xgi.get(player['id'], 0)),
        'XGI Previous GW': float(previous_gw_xgi.get(player['id'], 0)),
        'ΔGI': float(current_gw_xgi.get(player['id'], 0)) - float(previous_gw_xgi.get(player['id'], 0))
    }

    pdata['GI'] = pdata['Goals'] + pdata['Assists']
    pdata['Delta G'] = pdata['Goals'] - pdata['XG']
    pdata['Delta GI'] = pdata['GI'] - pdata['XGI']

    # Add upcoming GW columns dynamically
    for gw in next_5_gameweeks:
        gw_id = gw['id']
        gw_name = f"GW {gw_id}"
        pdata[gw_name] = ', '.join(team_opponents[player['team']][gw_id]['home_away'])
        pdata[f"{gw_name} Difficulty"] = ', '.join(str(d) for d in team_opponents[player['team']][gw_id]['difficulty'])
    
    player_info.append(pdata)

player_df = pd.DataFrame(player_info)

# ================= CALCULATE DIFFICULTY SCORE & FD INDEX =================
for gw in next_5_gameweeks:
    diff_col = f"GW {gw['id']} Difficulty"
    player_df[diff_col] = player_df[diff_col].apply(parse_difficulty)

player_df['Difficulty Score'] = player_df[[f"GW {gw['id']} Difficulty" for gw in next_5_gameweeks]].sum(axis=1)
player_df['FD Index'] = player_df.apply(
    lambda r: round(float(r['Form']) / r['Difficulty Score'], 2) if r['Difficulty Score'] not in (0, np.nan) else np.nan,
    axis=1
)

# ================= CREATE TRANSFER PICK TABLES =================
def create_Gw_transfers_in_table(position_name):
    base_cols = [
        'Player Name','Availability', 'Team', 'Position', 'Cost', 'Form', 'FD Index', 'XG', 'Clean Sheets', 'Goals', 'Assists',
        'XG Current GW','XG Previous GW', 'ΔG_GW', 'Delta G', 'XA', 'Delta GI', 'XG/90', 'Ownership (%)', 'GW Points',
        'Expected points Next GW', 'Total Points', 'Difficulty Score', 'Current Gameweek', 'GW Transfers In', 'GW Transfers Out'
    ]
    # add GW opponent/difficulty columns dynamically
    gw_cols = []
    for gw in next_5_gameweeks:
        gw_cols.append(f"GW {gw['id']}")
        gw_cols.append(f"GW {gw['id']} Difficulty")
    final_cols = base_cols + gw_cols
    df = player_df[player_df['Position'] == position_name]
    selected_cols = [c for c in final_cols if c in df.columns]
    return df[selected_cols].sort_values(by='GW Transfers In', ascending=False)

goalkeepers_Gw_transfers_in = create_Gw_transfers_in_table('Goalkeeper')
defenders_Gw_transfers_in = create_Gw_transfers_in_table('Defender')
midfielders_Gw_transfers_in = create_Gw_transfers_in_table('Midfielder')
forwards_Gw_transfers_in = create_Gw_transfers_in_table('Forward')

# ================= TEAM TABLES =================
teams_data = data['teams']
teams_df = pd.DataFrame(teams_data)[['id','short_name','name','played','points','form']]
teams_df.rename(columns={'short_name':'team','name':'Team'}, inplace=True)
teams_df['Goals Scored'] = 0
teams_df['Goals Conceded'] = 0
teams_df['Games'] = 0
teams_df['Last 5 GW Results'] = ""
team_results = {tid: [] for tid in teams_df['id']}

for fixture in fixtures:
    if not fixture['finished']:
        continue
    home, away = fixture['team_h'], fixture['team_a']
    home_goals = fixture['team_h_score'] or 0
    away_goals = fixture['team_a_score'] or 0
    teams_df.loc[teams_df['id']==home,'Goals Scored'] += home_goals
    teams_df.loc[teams_df['id']==home,'Goals Conceded'] += away_goals
    teams_df.loc[teams_df['id']==away,'Goals Scored'] += away_goals
    teams_df.loc[teams_df['id']==away,'Goals Conceded'] += home_goals
    teams_df.loc[teams_df['id']==home,'Games'] +=1
    teams_df.loc[teams_df['id']==away,'Games'] +=1

    home_name = teams_df.loc[teams_df['id']==home,'team'].values[0]
    away_name = teams_df.loc[teams_df['id']==away,'team'].values[0]
    result = f"{home_name} {home_goals} - {away_goals} {away_name}"
    team_results[home].append(result)
    team_results[away].append(result)

teams_df['Last 5 GW Results'] = teams_df['id'].apply(lambda tid: ', '.join(team_results[tid][-5:]))
teams_df['Goals Scored/Game'] = (teams_df['Goals Scored'] / teams_df['Games'].replace(0, np.nan)).round(1)
teams_df['Goals Conceded/Game'] = (teams_df['Goals Conceded'] / teams_df['Games'].replace(0, np.nan)).round(1)
teams_df['Goal Difference'] = teams_df['Goals Scored'] - teams_df['Goals Conceded']

attacking_teams = teams_df.sort_values(by=['Goals Scored/Game','Goals Scored','Goal Difference'],ascending=[False,False,False])[[
    'Team','Games','Goals Scored/Game','Goals Conceded/Game','Goals Scored','Goals Conceded','Goal Difference','Last 5 GW Results']]
attacking_teams['Last Updated'] = pd.to_datetime('now')

defensive_teams = teams_df.sort_values(by=['Goals Conceded/Game','Goals Conceded'],ascending=[True,True])[[
    'Team','Games','Goals Conceded/Game','Goals Conceded','Goals Scored/Game','Goals Scored','Goal Difference','Last 5 GW Results']]
defensive_teams['Last Updated'] = pd.to_datetime('now')

# ================= STATUS/KEY METRICS =================
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
    ["xG (Expected Goals)", "Predicts goal-scoring potential based on shot quality", "Helps find players likely to score"],
    ["Delta G (Goals - xG)", "Reveals finishing efficiency or over/underperformance", "Spot clinical finishers or potential regressions"],
    ["xA (Expected Assists)", "Gauges assist potential from creative passing", "Pick players with high assist potential"],
    ["ΔG_gw (Current GW XG - Previous GW XG)", "Tracks attacking momentum week-to-week", "Positive = improving; Negative = declining"],
    ["Delta GI (Goal Involvements - Expected GI)", "Shows actual impact vs expected contribution", "Positive = outperforming; Negative = underperforming"]
]
status_code_df = pd.DataFrame(status_code_info)
status_code_df['Last Updated'] = pd.to_datetime('now')

# ================= WRITE TO GOOGLE SHEET =================
write_to_sheet(sheet, player_df, 'Player Data')
write_to_sheet(sheet, goalkeepers_Gw_transfers_in.head(5), 'Smart Picks - Goalkeepers')
write_to_sheet(sheet, defenders_Gw_transfers_in.head(10), 'Smart Picks - Defenders')
write_to_sheet(sheet, midfielders_Gw_transfers_in.head(15), 'Smart Picks - Midfielders')
write_to_sheet(sheet, forwards_Gw_transfers_in.head(5), 'Smart Picks - Forwards')
write_to_sheet(sheet, attacking_teams, 'Best Attacking Teams')
write_to_sheet(sheet, defensive_teams, 'Best Defensive Teams')
write_to_sheet(sheet, status_code_df, 'FPL Key Metrics Guide')
