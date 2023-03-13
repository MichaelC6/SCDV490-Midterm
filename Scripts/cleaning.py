#This function is the cleaning of my data to create nice
#properly titled columns focusing on header and compare
import pandas as pd
import os
def getDF():
    # Get path to data
    path = os.path.dirname(os.getcwd())

    # The two files we'll be using
    header = pd.read_csv(path+'/data/header.csv')
    compare = pd.read_csv(path+'/data/comparison.csv')


    #Merging into one data frame based off of game id
    merged = pd.merge(header, compare, on='game_id')

    #Unfortunately this is the only way to do this...
    #Just gonna extract all the features I want.
    home_team = merged['team_id_a_x']
    away_team = merged['team_id_b_x']

    home_score = merged['score_a']
    away_score = merged['score_b']

    referee_1 = merged['referee_1']
    referee_2 = merged['referee_2']
    referee_3 = merged['referee_3']

    home_fouls = merged['fouls_a']
    away_fouls = merged['fouls_b']

    home_timeouts = merged['timeouts_a']
    away_timeouts = merged['timeouts_b']

    home_score_Q1 = merged['score_quarter_1_a']
    home_score_Q2 = merged['score_quarter_2_a']
    home_score_Q3 = merged['score_quarter_3_a']
    home_score_Q4 = merged['score_quarter_4_a']

    away_score_Q1 = merged['score_quarter_1_b']
    away_score_Q2 = merged['score_quarter_2_b']
    away_score_Q3 = merged['score_quarter_3_b']
    away_score_Q4 = merged['score_quarter_4_b']

    home_d_rebounds = merged['defensive_rebounds_a']
    away_d_rebounds = merged['defensive_rebounds_b']

    home_o_rebounds = merged['offensive_rebounds_a']
    away_o_rebounds = merged['offensive_rebounds_b']

    home_to_starters = merged['turnovers_starters_a']
    home_to_bench = merged['turnovers_bench_a']

    away_to_starters = merged['turnovers_starters_b']
    away_to_bench = merged['turnovers_bench_b']

    home_steals_starters = merged['steals_starters_a']
    home_steals_bench = merged['steals_bench_a']

    away_steals_starters = merged['steals_starters_b']
    away_steals_bench = merged['steals_bench_b']

    home_assists_starters = merged['assists_starters_a']
    home_assists_bench = merged['assists_bench_a']

    away_assists_starters = merged['assists_starters_b']
    away_assists_bench = merged['assists_bench_b']

    home_points_starters = merged['points_starters_a']
    home_points_bench = merged['points_bench_a']

    away_points_starters = merged['points_starters_a']
    away_points_bench = merged['points_bench_a']

    df = pd.DataFrame(
        {'home_team': home_team,
         'away_team': away_team,
         'home_score': home_score,
         'away_score': away_score,
         'referee_1': referee_1,
         'referee_2': referee_2,
         'referee_3': referee_3,
         'home_fouls': home_fouls,
         'away_fouls': away_fouls,
         'home_timeouts': home_timeouts,
         'away_timeouts': away_timeouts,
         'home_score_Q1': home_score_Q1,
         'home_score_Q2': home_score_Q2,
         'home_score_Q3': home_score_Q3,
         'home_score_Q4': home_score_Q4,
         'away_score_Q1': away_score_Q1,
         'away_score_Q2': away_score_Q2,
         'away_score_Q3': away_score_Q3,
         'away_score_Q4': away_score_Q4,
         'home_d_rebounds': home_d_rebounds,
         'away_d_rebounds': away_d_rebounds,
         'home_o_rebounds': home_o_rebounds,
         'away_o_rebounds': away_o_rebounds,
         'home_to_starters': home_to_starters,
         'home_to_bench': home_to_bench,
         'away_to_starters': away_to_starters,
         'away_to_bench': away_to_bench,
         'home_steals_starters': home_steals_starters,
         'home_steals_bench': home_steals_bench,
         'away_steals_starters': away_steals_starters,
         'away_steals_bench': away_steals_bench,
         'home_assists_starters': home_assists_starters,
         'home_assists_bench': home_assists_bench,
         'away_assists_starters': away_assists_starters,
         'away_assists_bench': away_assists_bench,
         'home_points_starters': home_points_starters,
         'home_points_bench': home_points_bench,
         'away_points_starters': away_steals_starters,
         'away_points_bench': away_steals_bench,
         }
    )
    return df