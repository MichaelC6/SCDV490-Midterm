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
model = MLPClassifier(random_state=7)

# Training the model on the training data and labels
model.fit(X_train, y_train_class)

# Testing the model i.e. predicting the labels of the test data.
y_pred_class = model.predict(X_test)

# Evaluating the results of the model
accuracy = accuracy_score(y_test_class,y_pred_class)*100
confusion_mat = confusion_matrix(y_test_class,y_pred_class)

# Turn this into a dataframe
matrix_df = pd.DataFrame(confusion_mat)

# Colors
yellow = "#ffc425"
green = "#1b4a33"
green2 = "#0DB02B"

# Plot the result
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(16,9))

sns.set(font_scale=2)

sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")

labels = ('winner','loser')

# Formatting details here
# Set axis titles
ax.set_title('Confusion Matrix - Classifier')
ax.set_xlabel("Predicted Outcome", fontsize = 15)
ax.set_xticklabels(labels)
ax.set_ylabel("True Outcome", fontsize= 15)
ax.set_yticklabels(labels, rotation = 0)
plt.show()

regr = MLPRegressor(random_state=7,solver='lbfgs',activation='relu').fit(X_train, y_train_regr)

predictions = regr.predict(X_test)
score = regr.score(X_test, y_test_regr)

margin_prediction = predictions[:, 0] - predictions[:, 1]

margin_true = np.array(y_train_regr['home_score']) - np.array(y_train_regr['away_score'])

plt.figure(figsize=(16, 10))
plt.tight_layout()
plt.hist(margin_prediction, bins=50, color=green)
plt.xlabel("Margin of Victory Home Team (points)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(np.arange(-40,41,5))
plt.title("Regression Model Predictions Histogram", fontsize=25);
plt.show()

plt.figure(figsize=(16, 10))
plt.tight_layout()
plt.hist(margin_true, bins=50, color=green)
plt.xlabel("Margin of Victory Home Team (points)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(np.arange(-40,41,5))
plt.title("Testing Data Histogram", fontsize=25);
plt.show()

plt.figure(figsize=(16, 10))
plt.tight_layout()
plt.hist(margin_true, bins=50, color=green, label='True Outcomes')
plt.hist(margin_prediction, bins=50, color=yellow, alpha=0.5, label='Predicted Outcomes')
plt.xlabel("Margin of Victory Home Team (points)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(np.arange(-40,41,5))
plt.legend()
plt.title("Testing Data Histogram", fontsize=25);
plt.show()

#Doing uncertainties
nsamples = 10000
sample_means = []
# Generate our 10000 samples
for n in range(nsamples):
    # Pick 10000 random numbers from our original dataset, allowing for repeats
    sample = np.random.choice(margin_prediction, len(margin_prediction), replace=True)
    # Calculate the mean and keep track of it
    sample_means.append(np.mean(sample))

mean_prediction = np.mean(sample_means)
std_prediction = np.std(sample_means)

print(f"error mean: {mean_prediction}")
print(f"error std: {std_prediction}")

#percent of scores within 2 standard deviations

above_zero = []

for item in margin_prediction:
    two_std = std_prediction * 2
    low = item - two_std
    high = item + two_std

    if low > 0 or high < 0:
        above_zero.append(1)
    else:
        above_zero.append(0)

unique, counts = np.unique(above_zero, return_counts=True)

plt.figure(figsize=(14, 14))
plt.tight_layout()
plt.bar(unique, counts, color=green, label='True Outcomes')
plt.xlabel("Conclusive?", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(unique,labels=('Not Conclusive','Conclusive'))
plt.title("Was the model able to predict a conclusive outcome?", fontsize=25);
plt.show()