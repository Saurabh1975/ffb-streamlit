import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import streamlit as st
import base64
from sklearn.linear_model import LinearRegression
from datetime import date


#Read In Data
df_proj = pd.read_csv('Data/fpros_projections.csv')
df_proj_floor = pd.read_csv('Data/fpros_projections_floor.csv')
df_proj_ceil = pd.read_csv('Data/fpros_projections_ceil.csv')


st.set_page_config(page_title="2023 Fantasy 🏈 Cheatsheet", layout="wide")



#Font Styles
section_title_style = 'font-size: 14; text-align: left; font-family: Roboto Black'
section_body_style = 'font-size: 14; text-align: left; font-family: Roboto'
title_style = 'font-size: 32px; font-weight: bold;font-family: Roboto Black'

st.markdown(f'<p style="{title_style}">🏈 2023 Fantasy Footaball Charts & Cheatsheets</p>', unsafe_allow_html=True)

subtitle_text = 'by <a href="https://twitter.com/SaurabhOnTap" target="_blank" rel="noopener noreferrer">Saurabh Rane</a> | Data via  <a href="https://fantasypros.com/" target="_blank" rel="noopener noreferrer">FantasyPros</a>, Last Updated ' +  df_proj['updated_date'].iloc[0]
st.markdown(f'<p style="{section_body_style}">{subtitle_text}</p>', unsafe_allow_html=True)



def calc_fpts(df_proj):


    pass_yds_pts = 1/pass_yds_pts_input
    #pass_td_pts = 4
    #pass_int_pts = -1

    rush_yds_pts = 1/rush_rec_yards

    #rec_att_pts = 0
    rec_yds_pts = 1/rush_rec_yards
    
    #rush_rec_td_pts = 6

    #fumble_lost_pts = -2

    df_proj['fpts'] = df_proj.pass_yds*pass_yds_pts + df_proj.pass_td*pass_td_pts + df_proj.pass_int*pass_int_pts + \
                    df_proj.rec_att*rec_att_pts + df_proj.rec_yds*rec_yds_pts + df_proj.rec_td*rush_rec_td_pts +\
                    df_proj.rush_yds*rush_yds_pts + df_proj.rush_td*rush_rec_td_pts +\
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


#Set Up Tabs
tab_league_settings, tab_rankings, tab_charts = st.tabs(["⚙️ Settings", "🔢Rankings", "📊 Charts"])


####Settings Tab#####

tab_league_settings.write(f'<p style="{section_title_style}">Overview</p>', unsafe_allow_html=True)


tab_league_settings.write(f'<p style="{section_body_style}">The goal of this app is to generate a custom cheatsheet based on FantasyPros projections for your league. Insert your league settings below and toggle over to the Rankings for an exportable cheatsheet. The Charts tab showcases positonal upside/downside within each player group</p>', unsafe_allow_html=True)


tab_league_settings.write(f'<p style="{section_body_style}">If you have any feedback, questions, or feature requests please reach out via <a href="https://twitter.com/SaurabhOnTap" target="_blank" rel="noopener noreferrer">Twitter</a>. Features in the pipeline are: abiltiy to mark drafted players, custom baselines, & incorporating ECR as a way to asssess downside/upside.</p>', unsafe_allow_html=True)

tab_league_settings.write(f'<p style="{section_title_style}">Scoring Settings</p>', unsafe_allow_html=True)
tab_league_settings.write(f'<p style="{section_body_style}">Inesrt league scoring settings below</p>', unsafe_allow_html=True)


#League Scoring Settings
scoring_col1, scoring_col2, scoring_col3, scoring_col4 = tab_league_settings.columns(4)

with scoring_col1:
    pass_yds_pts_input = st.number_input(label = 'Passing Yards/Pt', min_value = 1, value = 25, step = 1)
    rush_rec_yards = st.number_input(label = 'Rush/Rec Yards/Pt', min_value = 1, value = 10, step = 1)


with scoring_col2:
    pass_td_pts = st.number_input(label = 'Pass TD', min_value = 1, value = 4, step = 1)
    rush_rec_td_pts = st.number_input(label ='Rush/Rec TD', min_value = 1, value = 6, step = 1)



with scoring_col3:
    pass_int_pts = st.number_input(label = 'INTs', min_value = -6, value = -1, step = 1)
    rec_att_pts = st.number_input(label ='PPR', min_value = 0.0, value = 0.0, step = .25)



with scoring_col4:
    fumble_lost_pts = st.number_input(label = 'Fumbles', min_value = -6, value = -2, step = 1)
    
    
    
    
tab_league_settings.write(f'<p style="{section_title_style}">Roster Settings</p>', unsafe_allow_html=True)
tab_league_settings.write(f'<p style="{section_body_style}">Inesrt league roster settings below</p>', unsafe_allow_html=True)



roster_col1, roster_col2, roster_col3 = tab_league_settings.columns(3)


with roster_col1:
    teams = st.number_input(label = 'Teams', min_value = 1, value = 10, step = 1)
    wr = st.number_input(label = 'WR', min_value = 1, value = 3, step = 1)
    k = st.number_input(label = 'K', min_value = 0, value = 1, step = 1)
    budget = st.number_input(label = 'Auction Budget', min_value = 0, value = 200, step = 50)

with roster_col2:
    qb = st.number_input(label = 'QB', min_value = 1, value = 1, step = 1)
    te = st.number_input(label = 'TE', min_value = 1, value = 1, step = 1)
    dst = st.number_input(label = 'DST', min_value = 0, value = 1, step = 1)

with roster_col3:
    rb = st.number_input(label = 'RB', min_value = 1, value = 2, step = 1)
    flex = st.number_input(label = 'Flex (WR/RB)', min_value = 0, value = 1, step = 1)
    bench = st.number_input(label = 'Bench', min_value = 0, value = 5, step = 1)


starter_budget = budget - k - dst - bench
available_dollars = teams*starter_budget
    



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


df_proj = df_proj[['player', 'team', 'position', 'fpts', 'fpts_ceil', 'fpts_floor']]

df_proj = df_proj.sort_values(by=['position', 'fpts'], ascending=[True, False])

# Assign ranks within each position group & add fields

df_proj['voltailty'] =  df_proj.fpts_ceil - df_proj.fpts_floor
df_proj['fpts_upside'] =  df_proj.fpts_ceil - df_proj.fpts
df_proj['fpts_downside'] =  df_proj.fpts_floor - df_proj.fpts

df_proj['pos_rank'] = df_proj.groupby('position').cumcount() + 1



#Establish VBD
def assign_cutoff(row):
    if row['position'] == 'QB':
        return qb_cutoff
    elif row['position'] == 'RB':
        return rb_cutoff
    elif row['position'] == 'WR':
        return wr_cutoff
    elif row['position'] == 'TE':
        return te_cutoff
    else:
        return None

#Estbalish Value above FA
def assign_fa_cutoff(row):
    if row['position'] == 'QB':
        return qb_bench_cutoff
    elif row['position'] == 'TE':
        return te_bench_cutoff
    else:
        return flex_bench_cutoff
    
df_proj = df_proj.sort_values(by='fpts', ascending=False)



df_rb_proj = df_proj[(df_proj.position == 'RB') & (df_proj['pos_rank'] <= teams*(rb+flex))]
df_wr_proj = df_proj[(df_proj.position == 'WR') & (df_proj['pos_rank'] <= teams*(wr+flex))]
df_flex = pd.concat([df_rb_proj, df_wr_proj], ignore_index=True).sort_values(by=['fpts'], ascending=False)

#Bench Cutoffs
flex_cutoff = df_flex.iloc[teams*(wr + rb + flex) - 1].fpts
qb_cutoff = df_proj[df_proj.position == 'QB'].iloc[teams*qb ].fpts
rb_cutoff = max(df_proj[df_proj.position == 'RB'].iloc[teams*(rb+flex)].fpts,\
                min(df_proj[df_proj.position == 'RB'].iloc[teams*rb].fpts, flex_cutoff))
wr_cutoff =  max(df_proj[df_proj.position == 'WR'].iloc[teams*(wr+flex) ].fpts,\
                min(df_proj[df_proj.position == 'WR'].iloc[teams*wr].fpts, flex_cutoff))
te_cutoff = df_proj[df_proj.position == 'TE'].iloc[teams*te].fpts


qb_cutoff_disp = df_proj[df_proj.position == 'QB'].iloc[teams*qb]
rb_cutoff_disp = df_proj[df_proj.fpts == rb_cutoff].iloc[0]
wr_cutoff_disp = df_proj[df_proj.fpts == wr_cutoff].iloc[0]
te_cutoff_disp = df_proj[df_proj.position == 'TE'].iloc[teams*te ]


#Free Agent Cutoffs
qb_bench_cutoff = df_proj[df_proj.position == 'QB'].iloc[round(teams*qb*1.25) ].fpts
te_bench_cutoff = df_proj[df_proj.position == 'TE'].iloc[round(teams*te*1.25)].fpts
flex_bench_cutoff = df_proj[df_proj["position"].isin(["WR", "RB"])].iloc[round(rb+wr+flex+bench-2)*teams].fpts


qb_fa_cutoff_disp = df_proj[df_proj.fpts == qb_bench_cutoff].iloc[0]
te_fa_cutoff_disp = df_proj[df_proj.fpts == te_bench_cutoff].iloc[0]
flex_fa_cutoff_disp = df_proj[df_proj.fpts == flex_bench_cutoff].iloc[0]


    
df_proj['cutoff'] = df_proj.apply(assign_cutoff, axis=1)
df_proj['fa_cutoff'] = df_proj.apply(assign_fa_cutoff, axis=1)

df_proj['fpts_vbd'] = df_proj['fpts'] - df_proj['cutoff']
df_proj['fpts_vbd_FA'] = df_proj['fpts'] - df_proj['fa_cutoff']




####Rankings Tab#####



qb_cutoff_disp = df_proj[df_proj.position == 'QB'].iloc[teams*qb]
rb_cutoff_disp = df_proj[df_proj.fpts == rb_cutoff].iloc[0]
wr_cutoff_disp = df_proj[df_proj.fpts == wr_cutoff].iloc[0]
te_cutoff_disp = df_proj[df_proj.position == 'TE'].iloc[teams*te]


df_bench_cutoff = {
    "Position": ["QB", "RB", "WR", "TE"],
    "Player": [qb_cutoff_disp.player, rb_cutoff_disp.player, wr_cutoff_disp.player,  te_cutoff_disp.player],
    "Ranking": ["QB" + str(qb_cutoff_disp.pos_rank), 
                "WR" + str(rb_cutoff_disp.pos_rank), 
                "RB" + str(wr_cutoff_disp.pos_rank), 
                "TE" + str(te_cutoff_disp.pos_rank)],
    "Cutoff": [round(qb_cutoff), round(rb_cutoff), round(wr_cutoff), round(te_cutoff)]
}
df_bench_cutoff = pd.DataFrame(df_bench_cutoff)



df_fa_cutoff = {
    "Position": ["QB", "TE", "Flex"],
    "Player": [qb_fa_cutoff_disp.player, te_fa_cutoff_disp.player, flex_fa_cutoff_disp.player],
    "Ranking": ["QB" + str(qb_fa_cutoff_disp.pos_rank), 
                "TE" + str(te_fa_cutoff_disp.pos_rank), 
                flex_fa_cutoff_disp.position + str(flex_fa_cutoff_disp.pos_rank)],
    "Cutoff": [round(qb_bench_cutoff), round(te_bench_cutoff), round(flex_bench_cutoff)]
}
df_fa_cutoff = pd.DataFrame(df_fa_cutoff)




def highlight_position(row):
    color_map = {
        "QB": 'background-color: #E8BEDD',
        "RB": 'background-color: #DBC98F',
        "WR": 'background-color: #A5DEA4',
        "TE": 'background-color: #9AA5D9'
        # Add more color mappings for other positions
    }
    position = row['Pos.']
    return [color_map.get(position, '')] * len(row)



cutoff_desc_col1, cutoff_desc_col2,cutoff_desc_col3  = tab_rankings.columns(3)

cutoff_col1, cutoff_col2, cutoff_col3 = tab_rankings.columns(3)


with cutoff_desc_col1:
    st.write(f'<p style="{section_title_style}">Value Over Bench Cut-Offs</p>', unsafe_allow_html=True)
    st.write(f'<p style="{section_body_style}">Value over bench represents how many fantasy points a player is projected to score over the best bench player in that position</p>', unsafe_allow_html=True)
    
with cutoff_desc_col2:
    st.write(f'<p style="{section_title_style}">Value Over FA  Cut-Offs</p>', unsafe_allow_html=True)
    st.write(f'<p style="{section_body_style}">Value over FA represents how many fantasy points a player is projected to score over the best FA in that position. 25% of the league is assumed to carry back up QBs and TEs. The last two bench positions are assumed to be fungible with the next avaiable FA</p>', unsafe_allow_html=True)

with cutoff_desc_col3:
    st.write(f'<p style="{section_title_style}">Other Key Terms</p>', unsafe_allow_html=True)

    st.write(f'<p style="{section_body_style}"><b>Floor/Ceiling Above Expected:</b> Each player is expected to have a ceiling/floor based on their projected. For example, players proejcted to score 384 points, are expected to have a ceiling of 403 points. So Mahommes ceiling for 421 represents "+18" over expected. Floor/Ceiling values are calcualted using the low/high projections on FantasyPros.</p>', unsafe_allow_html=True)


with cutoff_col1: 
    st.dataframe(df_bench_cutoff)



with cutoff_col2:

    st.dataframe(df_fa_cutoff)


with cutoff_col3:

    st.write(f'<p style="{section_body_style}"><b>Setting Baseline:</b> By default player rankigns and auction values are calcualted using the best available bench player as a baseline (e.g. in a 10 team, 1 QB  league, QB11 is the best bench QB). I personally prefer this ranking method since it puts increased importance on starter level players and less on depth. If you prefer to baseline against hte worst FA avialable, use the radio buttons below.</p>', unsafe_allow_html=True)



df_filter_col1, df_filter_col2, df_filter_col3 = tab_rankings.columns(3)

with df_filter_col1:
    vbd_select = st.radio(
        "How do you want to determine baseline for auction values?",
        ["Best Bench Player", "Best FA"])
with df_filter_col2:
    position_filter = st.selectbox("Select Position:", ['All', 'QB', 'RB', 'WR', 'TE', 'Flex'])
with df_filter_col3:
    player_search = st.text_input("Search Player:")




if vbd_select == 'Best Bench Player':
    df_proj['proj_cost'] = round(df_proj.fpts_vbd/sum(df_proj[df_proj.fpts_vbd > 0].fpts_vbd) * available_dollars)
    baseline_string = 'Value Over Bench'
    baseline_label = 'Bench Baseline'

else:
    df_proj['proj_cost'] = round(df_proj.fpts_vbd_FA/sum(df_proj[df_proj.fpts_vbd_FA > 0].fpts_vbd_FA) * available_dollars)
    baseline_string = 'Value Over Bench'
    baseline_label = 'Bench Baseline'

df_proj['proj_cost'] = df_proj['proj_cost'].apply(lambda x: x if x >= 1 else 1)



#'fpts_floor', 'fpts_ceil'
df_display = df_proj[['player', 'team', 'position', 'pos_rank', 'fpts','fpts_vbd', 'fpts_downside', 'fpts_upside', 'fpts_vbd_FA', 'proj_cost']]

df_display.fillna(0, inplace=True)


# Filter the DataFrame based on positions and fpts_vbd
qb_display = df_display[df_display["position"] == "QB"].nlargest(round(qb*teams*1.25), "fpts_vbd")
te_display = df_display[df_display["position"] == "TE"].nlargest(round(te*teams*1.25), "fpts_vbd")
flex_display = df_display[df_display["position"].isin(["WR", "RB"])].nlargest(round((rb+wr+flex+bench)*teams*1.1), "fpts_vbd")




# Combine the filtered DataFrames
df_display = pd.concat([qb_display, te_display, flex_display])
df_display = df_display.sort_values(by='fpts_vbd', ascending=False).reset_index()


from sklearn.linear_model import LinearRegression


# Create a linear regression model
model = LinearRegression()
# Fit the model
X = df_display[['fpts']]
y = df_display['fpts_upside']
model.fit(X, y)

df_display['fpts_upside_error'] = model.predict(X) - df_display['fpts_upside'] 


# Create a linear regression model
model = LinearRegression()
# Fit the model
X = df_display[['fpts']]
y = df_display['fpts_downside']
model.fit(X, y)
df_display['fpts_downside_error'] = model.predict(X) - df_display['fpts_downside'] 



df_display = df_display.rename(columns={"player": "Player",
                                     "team": "Team",
                                      "position": "Pos.",
                                        "pos_rank": "Pos. Rank",
                                     "fpts": "Fantasy Points",
                                      "fpts_upside_error": "Ceiling vs Exp",
                                    "fpts_downside_error": "Floor vs Exp", 
                                    'fpts_vbd': 'Value Over Bench',
                                    'fpts_vbd_FA': 'Value Over FA',
                                       'proj_cost': "Projected Cost"})






df_display["Fantasy Points"] = df_display["Fantasy Points"].apply(lambda x: round(x))
df_display["Value Over Bench"] = df_display["Value Over Bench"].apply(lambda x: round(x))
df_display["Value Over FA"] = df_display["Value Over FA"].apply(lambda x: round(x))

df_display["Ceiling vs Exp"] = df_display["Ceiling vs Exp"].apply(lambda x: round(x, 1))
df_display["Floor vs Exp"] = df_display["Floor vs Exp"].apply(lambda x: round(x, 1))
df_display["Projected Cost"] = df_display["Projected Cost"].apply(lambda x: round(x))

df_display[ 'Pos.'] = df_display[ 'Pos.'] + df_display['Pos. Rank'].astype(str)


df_chart = df_display[:]

df_display = df_display[['Player', 'Team', 'Pos.','Fantasy Points', 'Ceiling vs Exp', 'Floor vs Exp', 
                         'Value Over Bench', 'Value Over FA', 'Projected Cost']]




# Add a download button
def download_csv():
    csv = df_display.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ffb-export.csv">Download CSV</a>'
    return href


# Function to apply styles based on conditions
def custom_style(val, column):
    if column == 'Pos.':
        if 'QB' in val:
            return 'background-color: #E8BEDD'
        elif 'RB' in val:
            return 'background-color: #DBC98F'
        elif 'WR' in val:
            return 'background-color: #A5DEA4'
        elif  'TE' in val:
            return 'background-color: #9AA5D9'     
    if column in ['Player', 'Value over Bench', 'Projected Cost']:
            return 'font-weight: 800;'

        
def background_gradient(val):
    max_val = df_display['Ceiling vs Exp'].max()
    min_val = df_display['Ceiling vs Exp'].min()
    normalized_val = (val - min_val) / (max_val - min_val)

    
    if val < 0:
        normalized_val = (-val - min_val) / (max_val - min_val)

        r = int(255 - normalized_val * (255 - 0xE3))  # White to e37c80 for negative values
        g = int(255 - normalized_val * (255 - 0x7C))
        b = int(255 - normalized_val * (255 - 0x80))
    else:

        r = int(255 - normalized_val * (255 - 0x67))  # White to 67a793 for positive values
        g = int(255 - normalized_val * (255 - 0xA7))
        b = int(255 - normalized_val * (255 - 0x93))
    
    return f'background-color: rgb({r}, {g}, {b})'
        


    

tab_rankings.markdown(download_csv(), unsafe_allow_html=True)

    
# Player Search

# Apply filters and search
filtered_df = df_display
if position_filter != 'All':
    if position_filter != 'Flex':
        filtered_df = filtered_df[filtered_df['Pos.'].str.contains(position_filter)]
    else:
        filtered_df = filtered_df[filtered_df['Pos.'].str.contains('WR|RB')]
if player_search:
    filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False)]
    

# Apply styles using the Styler object
styled_df = filtered_df.style.applymap(lambda x: custom_style(x, 'Pos.'), subset='Pos.') \
                    .applymap(lambda x: custom_style(x, 'Player'), subset='Player') \
                    .applymap(lambda x: custom_style(x, 'Team'), subset='Team') \
                    .applymap(lambda x: custom_style(x, 'Projected Cost'), subset='Projected Cost')\
                    .applymap(lambda x: background_gradient(x) if x is not None else '',
                              subset='Ceiling vs Exp') \
                    .applymap(lambda x: background_gradient(x) if x is not None else '',
                              subset='Floor vs Exp') \
                    .format({'Floor vs Exp': "{:+.1f}", 'Ceiling vs Exp': "{:+.1f}", 'Projected Cost': "${:.0f}"})



tab_rankings.dataframe(styled_df, 
                       height=35*50+38)



####Rankings Tab#####


#tab_charts





viz_df = df_chart[df_chart['Pos.'].str.contains('QB')]

def get_chart(viz_df, title_text):

    fig, ax = plt.subplots()



    y_val =  viz_df['Player']
    x_val =  viz_df['Fantasy Points']
    x_min =  viz_df['Fantasy Points'] + viz_df.fpts_downside
    x_max =  viz_df['Fantasy Points'] + viz_df.fpts_upside
    
    baseline =  viz_df[viz_df['Value Over Bench'] == 0]['Fantasy Points'].iloc[0]

    
    
    if len(viz_df) < 20:
        linewidth = 6
        labelsize = 6
    else:
        linewidth = 3
        labelsize = 4

    ax.axvline(x = baseline, linewidth = 0.5, color = "#404040", linestyle='dotted')
    
    ax.annotate('Starter Baseline', (baseline, 1), textcoords="offset points", xytext=(-1,0), ha='right', va = 'center', fontsize = labelsize)


    ax.hlines(y= y_val, xmin= x_min, xmax= x_val, color='#de425b', linewidth=linewidth, alpha = 0.3)
    ax.hlines(y= y_val, xmin= x_val, xmax= x_max, color='#488f31', linewidth=linewidth, alpha = 0.3)
    


    ax.scatter(x = x_val, y = viz_df['Player'], color = 'black', marker = "|") 
    #ax.scatter(x = x_min, y = viz_df['Player'], color = '#de425b') 
    #ax.scatter(x = x_max, y = viz_df['Player'], color = '#488f31') 
    
    for i, label in enumerate(y_val):
        ax.annotate(label, (x_val.iloc[i], i), textcoords="offset points", xytext=(-.5,0), ha='right', va = 'center', fontsize = labelsize)


    ax.invert_yaxis()

    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlabel(xlabel = "Proj, Fantasy Points", fontname='Roboto', fontsize=8)
    ax.set_title(label = title_text, fontname='Roboto Black', fontsize=10, loc='left',fontweight = 800)
    
    
    # Remove top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove y-axis and y-axis ticks
    ax.yaxis.set_visible(False)

    return(fig)


def get_baseline_chart(viz_df, title_text):

    fig, ax = plt.subplots()



    y_val =  viz_df['Player']
    
    
    
    x_val =  viz_df[baseline_string]
    x_min =  viz_df[baseline_string] + viz_df.fpts_downside
    x_max =  viz_df[baseline_string] + viz_df.fpts_upside
    

    if len(viz_df) < 20:
        linewidth = 6
        labelsize = 6
    else:
        linewidth = 3
        labelsize = 4

    ax.axvline(x = 0, linewidth = 0.5, color = "#404040", linestyle='dotted')
    
    ax.annotate(baseline_label, (0, 1), textcoords="offset points", xytext=(-1,0), ha='right', va = 'center', fontsize = labelsize)


    ax.hlines(y= y_val, xmin= x_min, xmax= x_val, color='#de425b', linewidth=linewidth, alpha = 0.3)
    ax.hlines(y= y_val, xmin= x_val, xmax= x_max, color='#488f31', linewidth=linewidth, alpha = 0.3)
    


    #ax.scatter(x = x_val, y = viz_df['Player'], color = 'black', marker = "|") 

        # Create a colormap for different positions
    pos_colors = {'QB': '#E854C0', 'RB': '#DBB742', 'WR': '#38A336', 'TE': '#5F74D9'}  # Add more as needed

    # Create a scatter plot
    for idx, row in viz_df.iterrows():
        color = 'black'  # Default color
        for pos, col in pos_colors.items():
            if pos in row['Pos.']:
                color = col
                break
        ax.scatter(x=x_val[idx], y=row['Player'], color=color, marker='|')

    #ax.scatter(x = x_min, y = viz_df['Player'], color = '#de425b') 
    #ax.scatter(x = x_max, y = viz_df['Player'], color = '#488f31') 
    
    for i, label in enumerate(y_val):
        ax.annotate(label, (x_val.iloc[i], i), textcoords="offset points", xytext=(-.5,0), ha='right', va = 'center', fontsize = labelsize)


    ax.invert_yaxis()

    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlabel(xlabel = "Proj, Fantasy Points Above Bench Replacement", fontname='Roboto', fontsize=8)
    ax.set_title(label = title_text, fontname='Roboto Black', fontsize=10, loc='left',fontweight = 800)
    
    
    # Remove top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove y-axis and y-axis ticks
    ax.yaxis.set_visible(False)

    return(fig)

tab_charts.write(f'<p style="{section_body_style}">If charts do not load right away, toggle to a different tab and re-click on Charts</p>', unsafe_allow_html=True)


viz_col1, viz_col2 = tab_charts.columns(2)


with viz_col1:
    
        
    
    viz_df = df_chart.sort_values(by=baseline_string, ascending = False)
    viz_df = viz_df.iloc[:50]
    st.pyplot(get_baseline_chart(viz_df, title_text = "Top " + str(len(viz_df.index)) + " Players, Floor/Ceiling"))
    
    viz_df = df_chart[df_chart['Pos.'].str.contains('QB')]
    st.pyplot(get_chart(viz_df, title_text = "Top " + str(len(viz_df.index)) + " QBs, Floor/Ceiling"))
    
    
    viz_df = df_chart[df_chart['Pos.'].str.contains('WR')].sort_values(by='Fantasy Points', ascending = False)
    viz_df = viz_df.iloc[:teams*(wr+flex+1)]
    st.pyplot(get_chart(viz_df, title_text = "Top " + str(len(viz_df.index)) + " WRs, Floor/Ceiling"))

    
    
with viz_col2:
    
    
    viz_df = df_chart.sort_values(by=baseline_string, ascending = False)
    viz_df = viz_df.iloc[50:100]
    st.pyplot(get_baseline_chart(viz_df, title_text = "Next " + str(len(viz_df.index)) + " Players, Floor/Ceiling"))
    
    viz_df = df_chart[df_chart['Pos.'].str.contains('RB')].sort_values(by='Fantasy Points', ascending = False)
    viz_df = viz_df.iloc[:teams*(rb+flex+1)]
    st.pyplot(get_chart(viz_df, title_text = "Top " + str(len(viz_df.index)) + " RBs, Floor/Ceiling"))
    
    
    
    viz_df = df_chart[df_chart['Pos.'].str.contains('TE')]
    viz_df = viz_df.head(40)
    st.pyplot(get_chart(viz_df, title_text = "Top " + str(len(viz_df.index)) + " TEs, Floor/Ceiling"))  
    
        
    
    
    