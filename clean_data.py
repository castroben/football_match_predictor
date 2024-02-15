import pandas as pd

def extract_data(filepath):
    df = pd.read_csv(filepath, sep=',') # read .csv file into dataframe
    df = df.dropna(how='all') # remove any entries from dataframe that contain no information
    
    # drop corrupted features and features that do not add useful information
    df = df.drop(
        ['Votes_for_Home', 'Votes_for_Draw', 'Votes_for_Away', 'Year', 'Minute',
        'Total_Bettors', 'Bet_Perc_on_Home', 'Bet_Perc_on_Draw', 'Bet_Perc_on_Away',
        'Team_1', 'Team_2', 'Rank_1', 'Rank_2','Large_Diff_win', 'Won_out_of_6',
        'Odds_Home', 'Odds_Draw', 'Odds_Away', 'Country_1', 'Country_2',
        'Detail_H2H', 'Indices_home', 'Indices_draw', 'Indices_away'],
        axis=1
    )

    # Dividing dataframe into 5 top European leagues and rest of the world
    dfEngland = df.loc[(df['League_type_country'] == 'england') & (df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3)]
    dfSpain = df.loc[(df['League_type_country'] == 'spain') & (df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3)]
    dfGermany = df.loc[(df['League_type_country'] == 'germany') & (df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3)]
    dfItaly = df.loc[(df['League_type_country'] == 'italy') & (df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3)]
    dfFrance = df.loc[(df['League_type_country'] == 'france') & (df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3)]
    dfWorld = df.loc[
        (df['League_type_country'] != 'england')
        & (df['League_type_country'] != 'spain')
        & (df['League_type_country'] != 'germany')
        & (df['League_type_country'] != 'italy')
        & (df['League_type_country'] != 'france')
        & (df['League_type_country'] != 'international')
        & (df['League_type_country'] != 'world cup')
        & (df['Number_of_H2H_matches'] >= 6) 
        & (df['Jumps'] < 3)
    ]
    dfAll = df.loc[(df['Number_of_H2H_matches'] >= 6) & (df['Jumps'] < 3) & (df['League_type_country'] != 'international') & (df['League_type_country'] != 'world cup')]


    # DROP LEAGUE_TYPE_COUNTRY, NUMBER_OF_H2H_MATCHES, JUMPS
    dfEngland = dfEngland.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfSpain = dfSpain.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfGermany = dfGermany.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfItaly = dfItaly.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfFrance = dfFrance.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfWorld = dfWorld.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)
    dfAll = dfAll.drop(['League_type_country', 'Number_of_H2H_matches', 'Jumps'], axis=1)

    # print(dfEngland.shape)
    # print(dfSpain.shape)
    # print(dfGermany.shape)
    # print(dfItaly.shape)
    # print(dfFrance.shape)
    # print(dfWorld.shape)
    # print(dfAll.shape)

    dfEngland.to_csv('preprocessed_data/england.csv', index=False)
    dfSpain.to_csv('preprocessed_data/spain.csv', index=False)
    dfGermany.to_csv('preprocessed_data/germany.csv', index=False)
    dfItaly.to_csv('preprocessed_data/italy.csv', index=False)
    dfFrance.to_csv('preprocessed_data/france.csv', index=False)
    dfWorld.to_csv('preprocessed_data/world.csv', index=False)
    dfAll.to_csv('preprocessed_data/all.csv', index=False)
    print(dfAll.shape)
    print('\n')

def generate_label(row):
    if row['Results_1'] > row['Results_2']:
        return 'Home'
    elif row['Results_1'] < row['Results_2']:
        return 'Away'
    else:
        return 'Draw'

def generate_classification_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Goals_Diff_1', 'Goals_Diff_2', 'Team_1_Found', 'Team_2_Found'], axis=1)
    df = pd.get_dummies(df)
    print(df.shape)
    df['Result'] =  df.apply(lambda row: generate_label(row), axis=1)
    df = df.drop(['Results_1', 'Results_2'], axis=1)
    #print(df.shape)
    df.to_csv('classification_data/classification_' + filepath.split('/')[1], index=False)

def generate_regression_data(filepath):
    df = pd.read_csv(filepath)
    df1 = df.drop(
        ['League_Rank_2', 'Games_played_2',  'Points_2', 'Won_2', 'Draw_2', 'Lost_2',
        'Goals_Scored_2', 'Goal_Rec_2', 'Goals_Diff_1', 'Goals_Diff_2', 'Team_2_Found',
        'Win_Perc_2', 'Draw_Perc_2', 'Avrage_FT_Goal', 'Average_HT_Goal', 'Results_2'
        ], axis=1
    )
    df1.rename(columns={
        "League_Rank_1" : "League_Rank",
        "Games_played_1" : "Games_played",
        "Points_1": "Points",
        "Won_1": "Won",
        "Draw_1": "Draw",
        "Lost_1": "Lost",
        "Goals_Scored_1": "Goals_Scored",
        "Goals_Rec_1": "Goals_Rec",
        "Team_1_Found": "Team",
        "Win_Perc_1": "Win_Perc",
        "Draw_Perc_1": "Draw_Perc",
        "Results_1": "Results"
    }, inplace=True)
    df1['Status'] = 1
    
    df2 = df.drop(
        ['League_Rank_1', 'Games_played_1',  'Points_1', 'Won_1', 'Draw_1', 'Lost_1',
        'Goals_Scored_1', 'Goals_Rec_1', 'Goals_Diff_2', 'Goals_Diff_1', 'Team_1_Found',
        'Win_Perc_1', 'Draw_Perc_1', 'Avrage_FT_Goal', 'Average_HT_Goal', 'Results_1'
        ], axis=1
    )
    df2.rename(columns={
        "League_Rank_2" : "League_Rank",
        "Games_played_2" : "Games_played",
        "Points_2": "Points",
        "Won_2": "Won",
        "Draw_2": "Draw",
        "Lost_2": "Lost",
        "Goals_Scored_2": "Goals_Scored",
        "Goal_Rec_2": "Goals_Rec",
        "Team_2_Found": "Team",
        "Win_Perc_2": "Win_Perc",
        "Draw_Perc_2": "Draw_Perc",
        "Results_2": "Results"
    }, inplace=True)
    df2['Status'] = 0

    result = pd.concat([df1, df2], axis=0, ignore_index=True)
    result = result.drop(["Team"], axis=1)
    result = pd.get_dummies(result)
    result.insert(len(result.columns)-1, 'Goals', result.pop('Results'))

    result.to_csv('regression_data/regression_' + filepath.split('/')[1], index=False)

if __name__ == '__main__':
    filepath = './analystm_mode_1_v1.csv'
    extract_data(filepath)
    
    files = ['preprocessed_data/england.csv', 'preprocessed_data/spain.csv', 'preprocessed_data/germany.csv', 'preprocessed_data/italy.csv', 'preprocessed_data/france.csv', 'preprocessed_data/world.csv', 'preprocessed_data/all.csv']
    for file in files:
        generate_classification_data(file)
        generate_regression_data(file)
    