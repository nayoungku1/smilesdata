import numpy as np

def pearson_corr(y_true, y_pred):
    """
    B = clip(Pearson correlation(y_true, y_pred), 0, 1)

    Parameters
    ----------
    y_true : array‑like, shape (n_samples,)
    y_pred : array‑like, shape (n_samples,)

    Returns
    -------
    float
        Pearson r을 0~1 범위로 잘라낸 값 (B)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true와 y_pred의 길이가 다릅니다.")

    # 상관계수 r (–1 ≤ r ≤ 1)
    r = np.corrcoef(y_true, y_pred)[0, 1]

    # 0~1 로 clip
    B = np.clip(r, 0.0, 1.0)
    return B