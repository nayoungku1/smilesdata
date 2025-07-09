# 🧾 Submission Log Viewer
## Dacon - Boost up AI 2025 : 신약 개발 경진대회

### 📅 2025-07-09 05:40:55
- **File:** `baseline_submit.csv`
- **Public Score:** 0.4770415417
- **Notes:** baseline.ipynb을 이용한 submission

### 📅 2025-07-09 09:28:09
- **File:** `pycarat_baseline_submit.csv`
- **Public Score:** 0.6653761038
- **Notes:** `allDescriptorFeature.ipynb` -> `optuna_feature_selection.py`에서 trial 100 -> `pycarat_nrmse.py` 
- **Metric**: NRMSE

### 📅 2025-07-10 02:13:59	
- **File:** `pycarat_quantile_transformer_baseline_submit.csv`
- **Public Score:** **0.6654734718**
- **Notes:** `allDescriptorFeature.ipynb` -> `label_transform.ipynb` > `QuantileTransformer` ->`optuna_feature_selection.py`에서 trial 100 (`100+quantile_selected_features.csv`)-> `pycarat_nrmse.py` 
- **Metric**: NRMSE
- **y**: transformed by Quantile
- **run**:```bash conda run -n smiles python pycarat_nrmse.py \ 
    --X_train data/X_train_allDescriptors.csv \
    --y_train data/y_train_quantile.csv --y_train_transformer data/quantile_transformer.pkl \
    --X_test data/X_test_allDescriptors.csv  --feature_file 100+quantile_selected_features.csv ```


### 📅 2025-07-10 02:16:31	
- **File:** `pycarat_power_transformer_baseline_submit.csv`
- **Public Score:** 0.6588803708
- **Notes:** `allDescriptorFeature.ipynb` -> `label_transform.ipynb` > `PowerTransformer` ->`optuna_feature_selection.py`에서 trial 100 (`100+powered_selected_features.csv`)-> `pycarat_nrmse.py` 
- **Metric**: NRMSE
- **y**: transformed by power
- **run**:```bash conda run -n smiles python pycarat_nrmse.py \ 
    --X_train data/X_train_allDescriptors.csv \
    --y_train data/y_train_power.csv --y_train_transformer data/power_transformer.pkl \      
    --X_test data/X_test_allDescriptors.csv  --feature_file 100+powered_selected_features.csv ```
