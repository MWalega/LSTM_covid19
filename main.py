import torch
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from covid19Predictor import covid19Predictor


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

# train function
def train_model(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels
):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 60

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(train_data)

        loss = loss_fn(y_pred.float(), train_labels)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)

                test_loss = loss_fn(y_test_pred.float(), test_labels)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.eval(), train_hist, test_hist

# create model
model = covid19Predictor(
    input_dim=1,
    hidden_dim=512,
    seq_length=seq_length,
    num_layers=2
)

model, train_hist, test_hist = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)