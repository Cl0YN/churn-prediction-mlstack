import optuna
from sklearn.metrics import roc_auc_score
import multiprocessing
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def tune_catboost(X_train, y_train, X_test, y_test, n_trials=50):
    """Tune CatBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 6000, 8000),
            'depth': trial.suggest_int('depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'objective': trial.suggest_categorical('objective', ['Logloss']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
            'random_strength': trial.suggest_int('random_strength', 0, 100),
            'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_state': trial.suggest_categorical('random_state', [42]),
        }
        model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', l2_leaf_reg=50, **params)
        model.fit(X_train, y_train, verbose=False)
        val_preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, val_preds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=multiprocessing.cpu_count())
    print('**BEST TRIAL**')
    print('Best Score: ', study.best_value)
    print('**CatBoost Tuned Hyperparameters**')
    print(study.best_trial.params)
    return study.best_trial.params

def tune_xgboost(X_train, y_train, X_test, y_test, n_trials=100):
    """Tune XGBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [10000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.99, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 0.99, log=True),
            'eval_metric': trial.suggest_categorical('eval_metric', ['auc']),
            'gamma': trial.suggest_categorical('gamma', [0, 0.25, 0.5, 1.0]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'random_state': trial.suggest_categorical('random_state', [42]),
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        val_preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, val_preds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=multiprocessing.cpu_count())
    print('**BEST TRIAL**')
    print('Best Score: ', study.best_value)
    print('**XGBoost Tuned Hyperparameters**')
    print(study.best_trial.params)
    return study.best_trial.params

def tune_lgbm(X_train, y_train, X_test, y_test, n_trials=100):
    """Tune LightGBM hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'objective': 'binary',
            'n_estimators': trial.suggest_int('n_estimators', 4000, 20000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 5, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 64),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
            'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
            'cat_l2': trial.suggest_int('cat_l2', 1, 20),
            'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
            'random_state': 42,
            'subsample': 0.6,
            'subsample_freq': 1,
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0),
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, val_pred)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=multiprocessing.cpu_count())
    print('**BEST TRIAL**')
    print('Best Score: ', study.best_value)
    print('**LGBM Tuned Hyperparameters**')
    print(study.best_trial.params)
    return study.best_trial.params