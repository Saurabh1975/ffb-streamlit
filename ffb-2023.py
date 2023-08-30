import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


def calc_fpts(df_proj):


    pass_yds_pts = .04
    pass_td_pts = 4
    pass_int_pts = -1

    rush_yds_pts = .1

    rec_att_pts = 0
    rec_yds_pts = .1
    
    rush_red_td_pts = 6

    fumble_lost_pts = -2

    df_proj['fpts'] = df_proj.pass_yds*pass_yds_pts + df_proj.pass_td*pass_td_pts + df_proj.pass_int*pass_int_pts + \
                    df_proj.rec_att*rec_att_pts + df_proj.rec_yds*rec_yds_pts + df_proj.rec_td*rush_red_td_pts +\
                    df_proj.rush_yds*rush_yds_pts + df_proj.rush_td*rush_red_td_pts +\
                    df_proj.fumble_lost*fumble_lost_pts


    return(df_proj)


def clean_fantasy_pull(df_proj):
    


    # Columns to convert to float
    float_columns = ['pass_yds',  'rush_yds', 'rec_yds']

    for col in float_columns:
        df_proj[col] = df_proj[col].str.replace(',', '').astype(float)

    df_proj.fillna(0, inplace=True)



    df_proj = calc_fpts(df_proj).sort_values(by='fpts', ascending = False)
    
    return(df_proj)





#Read In Data
df_proj = pd.read_csv('Data/fpros_projections.csv')
df_proj_floor = pd.read_csv('Data/fpros_projections_floor.csv')
df_proj_ceil = pd.read_csv('Data/fpros_projections_ceil.csv')

#Data Clean
df_proj = clean_fantasy_pull(df_proj)

df_proj_floor = clean_fantasy_pull(df_proj_floor)
df_proj_floor['player'] = df_proj_floor['player'].str.replace('low', '')
df_proj_floor = df_proj_floor[['player', 'fpts']]
df_proj_floor.rename(columns={"fpts": "fpts_floor"}, inplace=True)

df_proj_ceil = clean_fantasy_pull(df_proj_ceil)
df_proj_ceil['player'] = df_proj_ceil['player'].str.replace('high', '')
df_proj_ceil = df_proj_ceil[['player', 'fpts']]
df_proj_ceil.rename(columns={"fpts": "fpts_ceil"}, inplace=True)


#Combine Data
df_proj = pd.merge(df_proj, df_proj_ceil, how = 'left', on = 'player' )
df_proj = pd.merge(df_proj, df_proj_floor, how = 'left', on = 'player' )


df_proj = df_proj[['player', 'position', 'fpts', 'fpts_ceil', 'fpts_floor']]



df_proj = df_proj.sort_values(by=['position', 'fpts'], ascending=[True, False])

# Assign ranks within each position group & add fields

df_proj['voltailty'] =  df_proj.fpts_ceil - df_proj.fpts_floor
df_proj['fpts_upside'] =  df_proj.fpts_ceil - df_proj.fpts
df_proj['fpts_downside'] =  df_proj.fpts_floor - df_proj.fpts

df_proj['pos_rank'] = df_proj.groupby('position').cumcount() + 1





#Set Up Tabs
tab_league_settings, tab_rankings = st.tabs(["Settings", "Rankings"])



#Rankings

tab_rankings.dataframe( df_proj , use_container_width = True)