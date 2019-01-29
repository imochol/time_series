from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.model import DropColumn, grid_search


def test_grid_search_mlp(data_train_test):
    mlp = make_pipeline(
        DropColumn('timeStamp'),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(4, ), solver='lbfgs')
    )
    parameters = {
        "mlpregressor__alpha": [1.e-1, 1., 10., 100.],
        'mlpregressor__learning_rate_init': [0.1, 1.0, 10., 100., 1000.],
    }
    grid_search(
        train_test_data=data_train_test,
        model=mlp,
        parameters=parameters
    )
