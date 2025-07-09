import numpy as np
from sklearn.metrics import mean_squared_error

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (y_true.max() - y_true.min())