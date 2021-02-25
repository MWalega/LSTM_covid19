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

# create data sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()