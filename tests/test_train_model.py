import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from model.model import DropColumn, train_model


def test_train_model_svr(data_train_test, model_file_name):
    model = make_pipeline(
        DropColumn('timeStamp'),
        StandardScaler(),
        SVR(C=1.e5, gamma=0.005, kernel='rbf'),
    )
    train_model(data_train_test, model, model_file_name)


@pytest.mark.skip("")
def test_train_model_mlp(data_train_test, model_file_name):
    model = make_pipeline(
        DropColumn('timeStamp'),
        StandardScaler(),
        MLPRegressor(alpha=1.e-4, learning_rate_init=10.)
    )
    train_model(data_train_test, model, model_file_name)
