import pytest

from model.data import split_data, prepare_data, plot_timestamp_data


def test_plot_data(data):
    plot_timestamp_data(data, 'timeStamp', ['targets'])


@pytest.mark.skip("")
def test_prepare_data(data):
    X, y = prepare_data(data)
    new_cols = ['month', 'day', 'hour']
    for col in new_cols:
        assert col in X.columns


@pytest.mark.skip("")
def test_split(data):
    X, y = data.drop('targets', axis=1), data['targets']
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
