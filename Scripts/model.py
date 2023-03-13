
#Import libraries
import pandas as pd
import sklearn
import os
from sklearn.neural_network import MLPRegressor

# Get path to data
path = os.path.dirname(os.getcwd())

# Load all the data needed
header = pd.read_csv(path+'/data/header.csv')

# It is now time to make a collosial dataset and it will require a lot of cleaning beforehand
header_X = header(columns = ['game','date','time','round','phase','season_code','team_a','team_b',
                                  'game_time','remaining_partial_time','capacity','score_extra_time_1_a',
                                  'score_extra_time_2_a','score_extra_time_3_a','score_extra_time_1_b',
                                  'score_extra_time_2_b','score_extra_time_3_b','score_a','score_b','game_id','coach_a','coach_b','w_id'])
#To resolve the problem of duplicates...
#This link helped lead me to the right direction to look at the correct documentation
#https://stackoverflow.com/questions/73667349/how-to-do-one-hot-encoding-of-two-similar-columns-into-one
#Helpful Link https://www.sharpsightlabs.com/blog/pandas-get-dummies/
header_X.dropna()
X = pd.get_dummies(header_X, columns=['team_id_a','team_id_b'])
X = pd.get_dummies(X, columns=['referee_1','referee_2','referee_3'])
X = pd.get_dummies(X, columns=['stadium'])


# grab my scores for each team so
Y = header[['score_a','score_b']]



# What I want to get from this is score_a and score_b... that is the target output.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.4, random_state = 42)


# I get results here BUT... they are flawed.
regr = MLPRegressor().fit(X_train, y_train)

predictions = regr.predict(X_test)
score = regr.score(X_test, y_test)



