import torch
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# set plot style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# set matplotlib
register_matplotlib_converters()

# Explore data
df = pd.read_csv('data/time_series_covid19_confirmed_global.csv')
df = df.iloc[:, 4:]

daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

# plot data
plt.plot(daily_cases)
plt.title('Daily cases')
plt.show()

# data preprocessing
test_data_size = 80

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))