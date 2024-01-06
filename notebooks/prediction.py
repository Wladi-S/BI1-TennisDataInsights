# Wie gut lässt sich das Ergebnis eines Matches vorhersagen? Welches sind die wichtigsten Features für eine solche Prognose?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


matches = pd.read_csv('../data/raw/atp_matches_till_2022.csv')

df = matches[['tourney_id', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num', 
        'winner_id', 'winner_seed', 'winner_entry', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
        'loser_id', 'loser_seed', 'loser_entry', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 
        'best_of', 'round',
        'winner_rank', 'loser_rank']]

df = df.drop(columns=['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry'])

# Convert the tourney_date column to a string
df['tourney_date'] = df['tourney_date'].astype(str)

# Create new columns for year, month, and day
df['year'] = df['tourney_date'].str[:4].astype(int)
df['month'] = df['tourney_date'].str[4:6].astype(int)
df['day'] = df['tourney_date'].str[6:].astype(int)
df = df.drop(columns=['tourney_date'])

hand_encoder = LabelEncoder()
df['winner_hand'] = hand_encoder.fit_transform(df['winner_hand'].astype(str))
df['loser_hand'] = hand_encoder.transform(df['loser_hand'].astype(str))

df['winner_ioc'] = LabelEncoder().fit_transform(df['winner_ioc'].astype(str))
df['loser_ioc'] = LabelEncoder().fit_transform(df['loser_ioc'].astype(str))

df['surface'] = LabelBinarizer().fit_transform(df['surface'].astype(str))
df['tourney_level'] = LabelEncoder().fit_transform(df['tourney_level'].astype(str))
df['tourney_id'] = LabelEncoder().fit_transform(df['tourney_id'].astype(str))
df['round'] = LabelEncoder().fit_transform(df['round'].astype(str))

# Create a new DataFrame where person1 is always the player with the higher rank
df_higher_ranked_player = df[df['winner_rank'] < df['loser_rank']].copy()
df_higher_ranked_player['person1_id'] = df_higher_ranked_player['winner_id']
df_higher_ranked_player['person1_hand'] = df_higher_ranked_player['winner_hand']
df_higher_ranked_player['person1_ht'] = df_higher_ranked_player['winner_ht']
df_higher_ranked_player['person1_ioc'] = df_higher_ranked_player['winner_ioc']
df_higher_ranked_player['person1_age'] = df_higher_ranked_player['winner_age']
df_higher_ranked_player['person1_rank'] = df_higher_ranked_player['winner_rank']

df_higher_ranked_player['person2_id'] = df_higher_ranked_player['loser_id']
df_higher_ranked_player['person2_hand'] = df_higher_ranked_player['loser_hand']
df_higher_ranked_player['person2_ht'] = df_higher_ranked_player['loser_ht']
df_higher_ranked_player['person2_ioc'] = df_higher_ranked_player['loser_ioc']
df_higher_ranked_player['person2_age'] = df_higher_ranked_player['loser_age']
df_higher_ranked_player['person2_rank'] = df_higher_ranked_player['loser_rank']

# For df_higher_ranked_player, person1 is the winner
df_higher_ranked_player['person1_wins'] = 1

# Create a new DataFrame where person1 is always the player with the lower rank
df_lower_ranked_player = df[df['winner_rank'] > df['loser_rank']].copy()
df_lower_ranked_player['person1_id'] = df_lower_ranked_player['loser_id']
df_lower_ranked_player['person1_hand'] = df_lower_ranked_player['loser_hand']
df_lower_ranked_player['person1_ht'] = df_lower_ranked_player['loser_ht']
df_lower_ranked_player['person1_ioc'] = df_lower_ranked_player['loser_ioc']
df_lower_ranked_player['person1_age'] = df_lower_ranked_player['loser_age']
df_lower_ranked_player['person1_rank'] = df_lower_ranked_player['loser_rank']

df_lower_ranked_player['person2_id'] = df_lower_ranked_player['winner_id']
df_lower_ranked_player['person2_hand'] = df_lower_ranked_player['winner_hand']
df_lower_ranked_player['person2_ht'] = df_lower_ranked_player['winner_ht']
df_lower_ranked_player['person2_ioc'] = df_lower_ranked_player['winner_ioc']
df_lower_ranked_player['person2_age'] = df_lower_ranked_player['winner_age']
df_lower_ranked_player['person2_rank'] = df_lower_ranked_player['winner_rank']

# For df_lower_ranked_player, person1 is the loser
df_lower_ranked_player['person1_wins'] = 0

# Concatenate the two DataFrames
df_new = pd.concat([df_higher_ranked_player, df_lower_ranked_player])

# Now, person1 is always the player with the higher rank and person2 is always the player with the lower rank
# person1_wins is the target variable indicating whether the higher ranked player (person1) wins or not

columns_to_drop = ['winner_id', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank',
                   'loser_id', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank']
df_new.drop(columns=columns_to_drop, inplace=True)

# replace person1_ht, person1_age, person2_ht, person2_age with mean values
df_new['person1_ht'].fillna(df_new['person1_ht'].mean(), inplace=True)
df_new['person1_age'].fillna(df_new['person1_age'].mean(), inplace=True)
df_new['person2_ht'].fillna(df_new['person2_ht'].mean(), inplace=True)
df_new['person2_age'].fillna(df_new['person2_age'].mean(), inplace=True)

# ml model

y = df_new['person1_wins']
X = df_new.drop(columns='person1_wins')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Call the classifier
RF_classifier = RandomForestClassifier(random_state=42)
#fit the data
RF_classifier.fit(X_train, y_train)
#predict 
RF_predictions = RF_classifier.predict(X_test)

print(f"Accuracy Trainingsdaten: {RF_classifier.score(X_train, y_train)}")
print(f"Accuracy Testdaten: {RF_classifier.score(X_test, y_test)}")