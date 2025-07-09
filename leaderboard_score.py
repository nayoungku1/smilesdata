import numpy as np

def leaderboard_score(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : array‑like, shape (n_samples,)
        실제 Inhibition(%) 값
    y_pred : array‑like, shape (n_samples,)
        모델이 예측한 Inhibition(%) 값

    Returns
    -------
    dict
        {
            'A':  Normalized RMSE,
            'B':  Pearson r (0~1로 clip),
            'score': 최종 점수
        }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true와 y_pred의 길이가 다릅니다.")

    # --- A: Normalized RMSE --------------------------------------------------
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    range_y = y_true.max() - y_true.min()
    if range_y == 0:
        raise ValueError("y_true의 값이 모두 동일해 range가 0입니다.")
    A = rmse / range_y                 # Normalized RMSE
    A_cap = np.minimum(A, 1.0)         # min(A, 1)

    # --- B: Pearson 상관계수 (0~1 clip) --------------------------------------
    # np.corrcoef은 ddof=0 기준이므로 이미지와 같은 공분산/표준편차 정의와 일치
    r = np.corrcoef(y_true, y_pred)[0, 1]    # -1 ≤ r ≤ 1
    B = np.clip(r, 0.0, 1.0)

    # --- 최종 Score -----------------------------------------------------------
    score = 0.5 * (1.0 - A_cap) + 0.5 * B

    return score
