# This is a file for doing more exploring of the teams.csv
# in order to get an idea of where the league is at

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# I like fivethirtyeight style but one can easily comment out.
plt.style.use('fivethirtyeight')
import os
import time

#Import Team Data
path = os.path.dirname(os.getcwd())

# teams.csv as a pandas dataframe.
df = pd.read_csv(path+'/data/teams.csv')

# Grouping by season / year.
df1 = df.groupby('season_code').mean(numeric_only=True)

# Getting numpy array of the years to use on graphs.
years = np.arange(2007,2023,1)

# Getting labels to have fun data exploration
labels = np.array(df1.columns)

# Graph colors which are from the Siena style guide
# https://www.siena.edu/siena-style-guide/visual-styles/
yellow = "#ffc425"
green = "#1b4a33"
green2 = "#0DB02B"

# THIS MAY CRASH YOUR PC.
# This is for just intro exploration, I took out the key stats
# And made the seperate graphs below so that there was what
# I wanted in a more presentable format so BE WARNED!
do_graph = False
if do_graph:
    for label in labels[21:]:
        plt.figure(figsize=(16,9))
        plt.bar(years,np.array(df1[label]), color=green)
        plt.xlabel("year")
        plt.ylabel(label + " per year")
        plt.title(label + " vs year");
        plt.show()
        time.sleep(1)

# used this link to help
# https://www.python-graph-gallery.com/line-chart-dual-y-axis-with-matplotlib
# This is the graph for three point percentage and three points attempted per game

# Create Figure and make two axis
fig, ax1 = plt.subplots(figsize=(16,9))
ax2 = ax1.twinx()

# Plot bar chart for the threes per year, line for percentage.
ax1.bar(years,np.array(df1['three_points_attempted_per_game']), color=green)
ax2.plot(years,np.array(df1['three_points_percentage']),color=green2)

# Set x label as year
ax1.set_xlabel("year", fontsize=20);

# Set y labels and disable the percentage grid as it looks wonky if enabled
ax1.set_ylabel('three points attempted per game', color=green, fontsize=20);
ax2.set_ylabel('three point percentage', color=green2, fontsize=20);
ax2.grid(False)

#Add a title and show the graph!
fig.suptitle("three pointers attempted and percentage", fontsize=35);
plt.show()

#Graph for points per game.
plt.figure(figsize=(16, 9))
plt.bar(years, np.array(df1['points_per_game']), color=green)
plt.xlabel("year", fontsize=20)
plt.ylabel("points per game", fontsize=20)
plt.title("points per game over the years", fontsize=35);
plt.show()

#Graph for assists per game.
plt.figure(figsize=(16, 9))
plt.bar(years, np.array(df1['assists_per_game']), color=green)
plt.xlabel("year", fontsize=20)
plt.ylabel("assists per game", fontsize=20)
plt.title("assists per game over the years", fontsize=35);
plt.show()

#Graph for steals per game.
plt.figure(figsize=(16, 9))
plt.bar(years, np.array(df1['steals_per_game']), color=green)
plt.xlabel("year", fontsize=20)
plt.ylabel("steals per game", fontsize=20)
plt.title("steals per game over the years", fontsize=35);
plt.show()