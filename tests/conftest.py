import pytest

from model.data import load_data, prepare_data, split_data


@pytest.fixture()
def inputs_file():
    return "data/inputs.csv"


@pytest.fixture()
def targets_file():
    return "data/targets.csv"

@pytest.fixture()
def model_file_name():
    return "model.pkl"


@pytest.fixture()
def data(inputs_file, targets_file):
    return load_data(inputs_file, targets_file)


@pytest.fixture()
def data_prepared(data):
    return prepare_data(data)


@pytest.fixture()
def data_train_test(data_prepared):
    X, y = data_prepared
    return split_data(X, y)
