import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import roc_curve, auc
import numpy as np

from cleaning import getDF

df = getDF()

X = df.drop(columns=['home_score','away_score'])

X = pd.get_dummies(X, columns=['home_team'])
X = pd.get_dummies(X, columns=['away_team'])
X = pd.get_dummies(X, columns=['referee_1','referee_2','referee_3'])

#0 means home winner
#1 means away winner

winner = []

i = 0
while i < len(df):
    if df['home_score'][i] > df['away_score'][i]:
        winner.append(0)
    else:
        winner.append(1)
    i += 1

Y = pd.DataFrame({'home_score': df['home_score'], 'away_score': df['away_score'],'winner': winner})

# Splitting the data into tst and train
# 60 - 40 Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=7)

y_train_class = y_train['winner']
y_test_class = y_test['winner']

y_train_regr = y_train[['home_score','away_score']]
y_test_regr = y_test[['home_score','away_score']]

# Making the Neural Network Classifier
model = MLPClassifier()

# Training the model on the training data and labels
model.fit(X_train, y_train_class)

# Testing the model i.e. predicting the labels of the test data.
y_pred_class = model.predict(X_test)

# Evaluating the results of the model
accuracy = accuracy_score(y_test_class,y_pred_class)*100
confusion_mat = confusion_matrix(y_test_class,y_pred_class)

# Turn this into a dataframe
matrix_df = pd.DataFrame(confusion_mat)

# Plot the result
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(16,9))

sns.set(font_scale=2)

sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")

labels = ('winner','loser')

# Formatting details here
# Set axis titles
ax.set_title('Confusion Matrix - MLP')
ax.set_xlabel("Predicted label", fontsize = 15)
ax.set_xticklabels(labels)
ax.set_ylabel("True Label", fontsize= 15)
ax.set_yticklabels(labels, rotation = 0)
plt.show()

regr = MLPRegressor().fit(X_train, y_train_regr)

predictions = regr.predict(X_test)
score = regr.score(X_test, y_test_regr)