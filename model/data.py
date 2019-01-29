import pandas as pd
from matplotlib import pyplot as plt


def load_data(inputs_file, targets_file):
    inputs = pd.read_csv(inputs_file)
    inputs["timeStamp"] = inputs["timeStamp"].apply(pd.to_datetime)
    targets = pd.read_csv(targets_file, skiprows=1, names=['targets'])
    return inputs.join(targets)


def plot_timestamp_data(data, x_col, y_cols):
    data = data.set_index(x_col).last('5D').reset_index()
    fig = plt.figure()
    fig.suptitle('Target variable', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('Time')
    ax.set_ylabel('Targets')
    for y_col in y_cols:
        ax.plot_date(x=data[x_col], y=data[y_col], xdate=True, ydate=False)
    plt.show()
    return fig


def plot_true_vs_preds(data, x_col, y_col):
    fig = plt.figure()
    fig.suptitle('Scatter Plot', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('True')
    ax.set_ylabel('Prediction')
    ax.scatter(x=data[x_col], y=data[y_col])
    plt.show()


def prepare_data(data):
    data["month"] = data["timeStamp"].apply(lambda x: x.month)
    data["day"] = data["timeStamp"].apply(lambda x: x.day)
    data["hour"] = data["timeStamp"].apply(lambda x: x.hour)
    for column in data.columns:
        try:
            data[column] = data[column].apply(float)
        except TypeError:
            pass
    return data.drop("targets", axis=1), data['targets']


def split_data(X, y, test_size=0.2):
    size = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-size, :], X.iloc[-size:, :]
    y_train, y_test = y.iloc[:-size], y.iloc[-size:]
    return X_train, X_test, y_train, y_test
