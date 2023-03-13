import pandas as pd

from cleaning import getDF
import numpy as np
df = getDF()

df = df.drop(columns=['home_team','away_team','referee_1','referee_2','referee_3',])
#
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html

features= df.columns.values.tolist()

feature_means = []
feature_std = []
for feature in features:
    nsamples = 10000
    sample_means = []

    # Generate our 10000 samples
    for n in range(nsamples):
        # Pick 10000 random numbers from our original dataset, allowing for repeats
        sample = np.random.choice(df[feature], len(df[feature]), replace=True)

        # Calculate the mean and keep track of it
        sample_means.append(np.mean(sample))

    feature_means.append(np.mean(sample_means))
    feature_std.append(np.std(sample_means))

df1 = pd.DataFrame({'features': features, 'mean': feature_means, 'standard deviation': feature_std})
