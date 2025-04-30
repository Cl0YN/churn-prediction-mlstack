import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data, split_data, upsample_minority_class, get_preprocessor
from stacking import generate_meta_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


file_path = '/content/drive/MyDrive/Mu_files/Telco_customer_churn.xlsx'
data = load_data(file_path)
data = preprocess_data(data)




X_train, X_test, y_train, y_test = split_data(data)


X_train, y_train = upsample_minority_class(X_train, y_train)

preprocessor = get_preprocessor(X_train)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


catboost_params = {
    'iterations': 7811, 'depth': 4, 'learning_rate': 0.004304863460645952, 'objective': 'Logloss',
    'colsample_bylevel': 0.06129575006890019, 'random_strength': 49, 'boosting_type': 'Ordered',
    'bootstrap_type': 'Bernoulli', 'random_state': 42
}
xgb_params = {
    'n_estimators': 10000, 'learning_rate': 0.0019508127219592345, 'max_depth': 8,
    'colsample_bytree': 0.3678004823824082, 'subsample': 0.24370833574530723, 'eval_metric': 'auc',
    'gamma': 1.0, 'reg_lambda': 50.0, 'random_state': 42
}
lgbm_params = {
    'n_estimators': 6325, 'learning_rate': 0.0016682594078796002, 'reg_alpha': 8.794227315330197,
    'reg_lambda': 4.980958065716933, 'num_leaves': 858, 'min_child_samples': 98, 'max_depth': 63,
    'colsample_bytree': 0.32025661764519053, 'cat_smooth': 10, 'cat_l2': 4, 'min_data_per_group': 166,
    'min_child_weight': 8.766840084937344
}

cat = CatBoostClassifier(**catboost_params)
xgb = XGBClassifier(**xgb_params)
lgbm = LGBMClassifier(**lgbm_params)


models = [lgbm, xgb, cat]
meta_train, meta_test = generate_meta_features(models, X_train_processed, y_train, X_test_processed)


meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_train, y_train)


final_predictions = meta_model.predict_proba(meta_test)[:, 1]


roc_auc = roc_auc_score(y_test, final_predictions)
print(f'ROC AUC для мета-модели: {roc_auc}')


fpr, tpr, _ = roc_curve(y_test, final_predictions)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()