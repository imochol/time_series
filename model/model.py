import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model.data import plot_true_vs_preds


class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')
        return x.drop(self.column, axis=1)


def train_model(data, model, model_file_name, plot=True):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    joblib.dump(model, model_file_name)
    print('Model saved in the file {}'.format(model_file_name))
    prediction_results((X_train, y_train), model, "TRAIN")
    test_preds = prediction_results((X_test, y_test), model, "TEST")
    if plot:
        X_test.insert(len(X_test.columns), 'predicted', test_preds)
        plot_true_vs_preds(
            X_test.join(y_test),
            'targets',
            'predicted'
        )
    return model


def prediction_results(data, model, title):
    X, y = data
    y_pred = model.predict(X)
    print_results(title, y, y_pred)
    return y_pred


def print_results(title, true_values, predictions):
    print()
    print(title, "Mean absolute error")
    print(mean_absolute_error(true_values, predictions))
    print()


def grid_search(train_test_data, model, parameters, plot=True):
    # list of parameters: model.get_params().keys())
    X_train, X_test, y_train, y_test = train_test_data

    time_cv = TimeSeriesSplit(n_splits=3).split(X_train)
    grid_search = GridSearchCV(
        model,
        parameters,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        cv=time_cv,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    estimator = grid_search.best_estimator_
    test_preds = estimator.predict(X_test)
    print('Found parameters', grid_search.best_params_)
    print_results("TRAIN", y_train, estimator.predict(X_train))
    print_results("TEST", y_test, test_preds)
    if plot:
        X_test.insert(len(X_test.columns), 'predicted', test_preds)
        plot_true_vs_preds(
            X_test.join(y_test),
            'targets',
            'predicted'
        )
    return estimator
