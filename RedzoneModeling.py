# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:49:22 2025

@author: NAWri
"""

import pandas as pd
import nfl_data_py as nfl
seasons = [2021, 2022, 2023]
pbp = nfl.import_pbp_data(seasons)
redzonepbp = pbp[pbp['yardline_100']<=20.0]
nfl.import_win_totals(seasons)
columns = ['posteam','desc','play_type','rush_attempt','pass_attempt','yards_gained',
           'goal_to_go','down','qtr','ydstogo','shotgun','no_huddle','qb_dropback',
           'run_location','run_gap','score_differential','td_prob',
           'fg_prob','wp','def_wp','touchdown','offense_formation','offense_personnel',
           'defenders_in_box','defense_personnel','number_of_pass_rushers',
           'route','defense_man_zone_type','defense_coverage_type','ngs_air_yards',
           'xpass','cp','game_id','two_point_attempt','sack']
redzonepbpfiltered = redzonepbp[columns]
redzonepbpfiltered.to_csv(("C:/Users/NAWri/Documents/BGA/Spring 2025/RedzoneModeling/redzonepbpfiltered.csv"),index=False)
redzonepct = pd.read_csv("C:/Users/NAWri/Documents/BGA/Spring 2025/RedzoneModeling/redzonepct.csv").dropna(axis=1, how='all')
#-------------------------------------------------------------------------------------------------------------------------
print(redzonepct.info())
deadplays = ['extra_point', 'no_play', 'field_goal','qb_kneel','kickoff','qb_spike']
redzoneactionplays = redzonepbpfiltered[~redzonepbpfiltered['play_type'].isin(deadplays)]
redzoneactionplays = redzoneactionplays[redzoneactionplays['score_differential'].abs() <= 16]
redzoneactionplays = redzoneactionplays[redzoneactionplays['two_point_attempt']==0]
#--------------------
filtered_redzoneaction = redzoneactionplays.copy()

condition = ~(
    (filtered_redzoneaction['play_type'] == 'pass') &
    (filtered_redzoneaction['sack'] != 1) & 
    (filtered_redzoneaction['ngs_air_yards'].isna()) & 
    (filtered_redzoneaction['route'].isna())
)

filtered_redzoneaction = filtered_redzoneaction[condition]
#-----------
# Statement to bucket air yard distances
import numpy as np

def categorize_play(row):
    if row['rush_attempt'] == 1:
        return 'run'
    elif row['sack'] == 1:
        return 'sack'
    elif pd.notna(row['ngs_air_yards']):
        if row['ngs_air_yards'] < 0:
            return 'passscreen'
        elif row['ngs_air_yards'] <= 10:
            return 'passshort'
        else:
            return 'passmedium+'
    else:
        return None

filtered_redzoneaction['play_category'] = filtered_redzoneaction.apply(categorize_play, axis=1)
filtered_redzoneaction = filtered_redzoneaction.dropna(subset=['play_category'])
filtered_redzoneaction.to_csv(("C:/Users/NAWri/Documents/BGA/Spring 2025/RedzoneModeling/filteredredzoneaction.csv"),index=False)
#----------------
team_name_to_abbr = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

redzonepct['posteam'] = redzonepct['posteam'].replace(team_name_to_abbr)
#--------------------------------------------------------------------------
def calculate_play_distributions(df, season):
    season_df = df[df['season'] == season]
    total_plays = season_df.groupby('posteam')['play_category'].count().rename('total_plays')
    play_counts = season_df.groupby(['posteam', 'play_category']).size().unstack(fill_value=0)
    play_dist = (play_counts.div(total_plays, axis=0) * 100).reset_index()
    play_dist['season'] = season

    redzonepct_filtered = redzonepct[redzonepct['Season'] == season][['posteam', 'RZPct']]
    play_dist = play_dist.merge(redzonepct_filtered, on='posteam', how='left')

    return play_dist

playdist2021 = calculate_play_distributions(filtered_redzoneaction, 2021)
playdist2022 = calculate_play_distributions(filtered_redzoneaction, 2022)
playdist2023 = calculate_play_distributions(filtered_redzoneaction, 2023)
#-----------
def stack_play_distribution_dfs(*dfs):
    return pd.concat(dfs, ignore_index=True)
stacked_playdist = stack_play_distribution_dfs(playdist2021, playdist2022, playdist2023)
stacked_playdist['RZPct'] = stacked_playdist['RZPct'].str.replace('%', '').astype(float) / 100
stacked_playdist.to_csv(("C:/Users/NAWri/Documents/BGA/Spring 2025/RedzoneModeling/playdists.csv"),index=False)
#-----------------------------------------------------------------------------
#comp % by air yards
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

air_yards_cp = filtered_redzoneaction.groupby('ngs_air_yards')['cp'].mean().reset_index().dropna()

slope, intercept, r_value, p_value, std_err = linregress(air_yards_cp['ngs_air_yards'], air_yards_cp['cp'])

plt.figure(figsize=(12, 8))
sns.lineplot(x='ngs_air_yards', y='cp', data=air_yards_cp, color='teal', label='Average Completion Probability')
plt.plot(air_yards_cp['ngs_air_yards'], intercept + slope * air_yards_cp['ngs_air_yards'], color='red', linestyle='--', label=f'Trend Line (Slope: {slope:.4f})')

plt.xlabel('NGS Air Yards')
plt.ylabel('Average Completion Probability (cp)')
plt.title('Average Completion Probability by NGS Air Yards (2021-2023)')
plt.legend()
plt.show()
#---------------------------------------------------------------------------------
playdists = pd.read_csv("C:/Users/NAWri/Documents/BGA/Spring 2025/RedzoneModeling/playdists.csv")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_league_play_distribution(playdists):
    play_types = ['run', 'passscreen', 'passshort', 'passmedium+']  # adjust if you renamed/added
    avg_by_season = playdists.groupby('season')[play_types].mean().reset_index()
    melted = avg_by_season.melt(id_vars='season', value_vars=play_types,
                                var_name='Play Type', value_name='Frequency')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='Play Type', y='Frequency', hue='season')

    plt.title('League-Wide Red Zone Playcall Distribution by Year')
    plt.ylabel('Percentage of Plays')
    plt.xlabel('Play Type')
    plt.legend(title='Season')
    plt.tight_layout()
    plt.show()

plot_league_play_distribution(playdists)
#--------------------------------------------------------------------------------------
from scipy.stats import pearsonr

def plot_playtype_correlations(df):
    play_types = ['run', 'passscreen', 'passshort', 'passmedium+']
    correlations = {}

    for pt in play_types:
        if df[pt].notna().all() and df['RZPct'].notna().all():
            corr, _ = pearsonr(df[pt], df['RZPct'])
            correlations[pt] = corr

    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    corr_df = corr_df.sort_values(by='Correlation')

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Correlation', y=corr_df.index, data=corr_df, palette='coolwarm')
    plt.title('Correlation Between Play Type Frequency and Red Zone TD%')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Play Type')
    plt.tight_layout()
    plt.show()

plot_playtype_correlations(playdists)
#---------------------------------------------------------------------------------------
def scatter_playtype_vs_rzpct(df, play_type):
    plt.figure(figsize=(7, 5))
    sns.lmplot(data=df, x=play_type, y='RZPct', hue='season', height=5, aspect=1.2)
    plt.title(f'{play_type} Frequency vs Red Zone TD%')
    plt.xlabel(f'{play_type} Play Frequency (%)')
    plt.ylabel('Red Zone TD%')
    plt.tight_layout()
    plt.show()

scatter_playtype_vs_rzpct(playdists, 'passshort')
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

route_cp = filtered_redzoneaction.groupby('route')['cp'].mean().sort_values()

plt.figure(figsize=(12, 8))
plt.barh(route_cp.index, route_cp.values, color='skyblue')
plt.xlabel('Average Completion Probability (cp)')
plt.ylabel('Route Type')
plt.title('Average Completion Probability by Route Type (2021-2023)')
plt.show()