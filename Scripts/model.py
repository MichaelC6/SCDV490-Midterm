
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os

# Get path to data
path = os.path.dirname(os.getcwd())

# Load all the data needed
header = pd.read_csv(path+'/data/header.csv')
compare = pd.read_csv(path+'/data/compare.csv')
boxscore = pd.read_csv(path+'/data/boxscore.csv')

# It is now time to make a collosial dataset and it will require a lot of cleaning beforehand

header_X = header.drop()
import sklearn.preprocessing as prep




