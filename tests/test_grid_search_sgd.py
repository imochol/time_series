from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from model.model import DropColumn, grid_search


def test_grid_search_sgd(data_train_test):
    parameters = {
        "sgdregressor__penalty": ["l2", "l1"],
        "sgdregressor__alpha": [1.e-4, 1.e-2, 1., 10.],
    }
    sgd = make_pipeline(
        DropColumn('timeStamp'),
        # StandardScaler(),
        PolynomialFeatures(),
        SGDRegressor(tol=1.e-4),
    )
    grid_search(
        train_test_data=data_train_test,
        model=sgd,
        parameters=parameters
    )
