#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Import Team Data
path = os.path.dirname(os.getcwd())

teams = pd.read_csv(path+'/data/teams.csv')

print("hello world")