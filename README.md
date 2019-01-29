Data should be copied to:
data/inputs.csv
data/targets.csv

To train the final model run from the main directory:
pytest tests/test_train_model.py

To make the grid search for SVR run from the main directory:
pytest tests/test_grid_search_svr.py

To make the grid search for MLP run from the main directory:
pytest tests/test_grid_search_mlp.py

The analysis and model searching is described in summary.pdf
