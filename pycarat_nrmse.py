# ---------------- Imports ----------------
import argparse
from typing import Callable, Optional, Tuple
import pandas as pd

# PyCaret import를 명시적으로 alias 해두는 편이 훨씬 안전
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare_models,
    tune_model as reg_tune_model, finalize_model as reg_finalize_model,
    predict_model as reg_predict_model, add_metric as reg_add_metric
)
from pycaret.classification import (
    setup as cls_setup, compare_models as cls_compare_models,
    tune_model as cls_tune_model, finalize_model as cls_finalize_model,
    predict_model as cls_predict_model, add_metric as cls_add_metric
)

from nrmse import nrmse   # ← 함수만 가져오기

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="PyCaret AutoML runner")
    p.add_argument("--X_train", required=True)
    p.add_argument("--y_train", required=True)
    p.add_argument("--X_test",  required=True)
    p.add_argument("--feature_file", required=True,
                   help="CSV with selected feature names (one col)")
    return p.parse_args()

# ---------------- Core ----------------
def pycaret_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    *,
    task: str = "regression",
    metric_id: str = "NRMSE",
    metric_func: Optional[Callable] = None,
    greater_is_better: bool = False,
    session_id: int = 777,
    normalize: bool = True
) -> Tuple[pd.Series, object]:

    train_df = X_train.copy()
    train_df["target"] = y_train.squeeze()          # Series 보장

    # 분류/회귀 스위치
    if task.lower().startswith("reg"):
        pc_setup, pc_compare, pc_tune, pc_finalize, pc_predict, pc_add_metric = (
            reg_setup, reg_compare_models, reg_tune_model,
            reg_finalize_model, reg_predict_model, reg_add_metric
        )
    else:
        pc_setup, pc_compare, pc_tune, pc_finalize, pc_predict, pc_add_metric = (
            cls_setup, cls_compare_models, cls_tune_model,
            cls_finalize_model, cls_predict_model, cls_add_metric
        )

    # 세션 초기화
    pc_setup(
        data=train_df,
        target="target",
        session_id=session_id,
        normalize=normalize,
        verbose=False
    )

    # 커스텀 메트릭 등록
    if metric_func is not None:
        pc_add_metric(
            id=metric_id, name=metric_id,
            score_func=metric_func,
            greater_is_better=greater_is_better
        )

    best   = pc_compare(sort=metric_id)
    tuned  = pc_tune(best, optimize=metric_id)
    final  = pc_finalize(tuned)

    preds  = pc_predict(final, data=X_test)["prediction_label"]

    print(f"[PyCaret] Best model ({metric_id}): {tuned.__class__.__name__}")
    print(f"[PyCaret] Params → {tuned.get_params()}")

    return preds, final

# ---------------- main ----------------
def main():
    args = parse_args()

    # 1) 데이터 로드
    X_train = pd.read_csv(args.X_train)
    y_train = pd.read_csv(args.y_train).squeeze()
    X_test  = pd.read_csv(args.X_test)
    feat_list = pd.read_csv(args.feature_file).squeeze().tolist()

    # 2) 동일 피처 서브셋 적용
    X_train = X_train[feat_list]
    X_test  = X_test[feat_list]

    # 3) 학습 & 예측
    y_pred, _ = pycaret_train(
        X_train, y_train, X_test,
        task="regression",
        metric_id="NRMSE",
        metric_func=nrmse,           # <‑ 함수 직접
        greater_is_better=False
    )

    # 4) 필요하면 저장
    
    submit = pd.read_csv('data/sample_submission.csv')
    submit['Inhibition'] = pd.Series(y_pred, name="prediction")
    submit.to_csv('submit/pycarat_baseline_submit.csv',index=False)

if __name__ == "__main__":
    main()
