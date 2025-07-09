#!/usr/bin/env python
# optuna_feature_selection.py
import argparse
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
import optuna.visualization as vis
from optuna.visualization.matplotlib import plot_optimization_history as plot_optimization_history_mpl
import matplotlib.pyplot as plt

# ---------------- nRMSE (범위 기반) ----------------
def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (y_true.max() - y_true.min())

# ---------------- CLI 인자 ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna‑based feature selection using XGBoost + nRMSE"
    )
    p.add_argument("--X_train", type=str, required=True,
                   help="CSV with shape (n_samples, n_features)")
    p.add_argument("--y_train", type=str, required=True,
                   help="CSV with shape (n_samples, 1)")
    p.add_argument("--trials", type=int, default=200,
                   help="Number of Optuna trials (default: 200)")
    p.add_argument("--output", type=str, required=True,
                   help="selected_feature prefix")
    return p.parse_args()

# ---------------- 메인 ----------------
def main():
    
    args = parse_args()

    # 1) 데이터 로드
    X_df = pd.read_csv(args.X_train)
    y_df = pd.read_csv(args.y_train)

    X = X_df.values.astype(np.float32)          # (n, p)
    y = y_df.squeeze().values.astype(np.float32)  # (n,)

    n_features = X.shape[1]
    RANDOM_STATE = 42
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    # 2) Optuna objective (X, y, cv 캡처)
    def objective(trial):
        # --- 이진 마스크 제안 ---
        mask = np.array(
            [trial.suggest_int(f"f{i}", 0, 1) for i in range(n_features)],
            dtype=bool,
        )
        if not mask.any():                      # 최소 한 개
            idx = trial.suggest_int("fallback_idx", 0, n_features - 1)
            mask[idx] = True

        X_sel = X[:, mask]

        # --- XGBoost (고정 하이퍼파라미터) ---
        model = XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        # --- CV 평가 ---
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_sel):
            model.fit(X_sel[tr_idx], y[tr_idx])
            preds = model.predict(X_sel[va_idx])
            fold_scores.append(nrmse(y[va_idx], preds))

        return np.mean(fold_scores)             # 최소화

    # 3) Optuna 실행
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # 4) 결과 요약 & 마스크 저장
    best_mask = np.array(
        [study.best_trial.params.get(f"f{i}", 0) for i in range(n_features)],
        dtype=bool,
    )
    print(f"\n▶ Best nRMSE       : {study.best_value:.5f}")
    print(f"▶ Selected features: {best_mask.sum()}/{n_features}")

    selected_cols = X_df.columns[best_mask]
    selected_cols.to_series().to_csv(f"{args.trials}+{args.output}_selected_features.csv", index=False)
    print(f"▶ Saved mask → {args.trials}+{args.output}_selected_features.csv")

    import os
    os.makedirs("optuna_plots", exist_ok=True)

    # Plotly interactive plots
    plotly_figs = {
        "optimization_history": vis.plot_optimization_history(study),
        "param_importance": vis.plot_param_importances(study),
        "parallel_coord": vis.plot_parallel_coordinate(study),
        "slice": vis.plot_slice(study),
        "edf": vis.plot_edf(study),
    }

    for name, fig in plotly_figs.items():
        fig.write_html(f"optuna_plots/{args.trials}trials_plot_{name}.html")
        print(f"Saved interactive plot: {args.trials}trials_plot_{name}.html")

    # Matplotlib static version (optional)
    fig_mpl = plot_optimization_history_mpl(study)
    fig_mpl.set_title("Optimization History (nRMSE)")
    plt.tight_layout()
    plt.savefig(f"optuna_plots/{args.trials}+{args.output}+trials_plot_optimization_history.png", dpi=300)
    print(f"Saved static plot: {args.trials}+{args.output}+trials_plot_optimization_history.png")

if __name__ == "__main__":
    main()
