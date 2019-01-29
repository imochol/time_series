from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from model.model import DropColumn, grid_search


def test_grid_search_svr(data_train_test):
    parameters = {
        "svr__kernel": ["rbf"],
        "svr__C": [1.e5],
        "svr__gamma": [5.e-4, 1.e-3, 5e-3]
    }
    svr = make_pipeline(
        DropColumn('timeStamp'),
        StandardScaler(),
        SVR(),
    )
    grid_search(
        train_test_data=data_train_test,
        model=svr,
        parameters=parameters
    )
