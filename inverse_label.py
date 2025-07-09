import joblib

def inverse_label(transformer, y_test_pred):
    """
    Inverse Transform the y_test
    """

    transformer = joblib.load(transformer)

    return transformer.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
