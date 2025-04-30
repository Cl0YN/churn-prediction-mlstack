import numpy as np
from sklearn.model_selection import StratifiedKFold

def generate_meta_features(models, X_train, y_train, X_test, n_folds=5):
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    n_models = len(models)
    meta_train = np.zeros((X_train.shape[0], n_models))
    meta_test = np.zeros((X_test.shape[0], n_models))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        for m_idx, model in enumerate(models):
            model.fit(X_tr, y_tr)
            meta_train[val_idx, m_idx] = model.predict_proba(X_val)[:, 1]
            meta_test[:, m_idx] += model.predict_proba(X_test)[:, 1] / n_folds

    return meta_train, meta_test