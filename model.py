import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from geopy.distance import great_circle


def load_and_preprocess_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['trans_date'] + ' ' + combined['trans_time'], errors='coerce')
    combined[['hour', 'day_of_week', 'day_of_month', 'month', 'year']] = combined['datetime'].apply(lambda dt: [dt.hour, dt.dayofweek, dt.day, dt.month, dt.year] if pd.notnull(dt) else [np.nan] * 5).apply(pd.Series)
    combined['dob'] = pd.to_datetime(combined['dob'], errors='coerce')
    combined['age'] = combined['year'] - combined['dob'].dt.year
    valid_coords = combined[['lat', 'long', 'merch_lat', 'merch_long']].notnull().all(axis=1)
    combined.loc[valid_coords, 'distance'] = combined.loc[valid_coords].apply(lambda row: great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km, axis=1)
    combined['amt_log'] = np.log1p(combined['amt'])
    combined['transaction_speed'] = combined['distance'] / (combined['amt'] + 1)
    combined['distance_per_amount'] = combined['distance'] / (combined['amt'] + 1)

    # encoding
    categorical_cols = ['category', 'gender', 'state', 'job', 'city', 'merchant']
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    combined['cc_num_txn_count'] = combined.groupby('cc_num')['id'].transform('count')

    # drop cols
    drop_cols = ['trans_date', 'trans_time', 'unix_time', 'first', 'last', 'street', 'zip', 'lat', 'long', 'merch_lat', 'merch_long', 'dob', 'datetime', 'trans_num']
    combined.drop(columns=[col for col in drop_cols if col in combined.columns], inplace=True)

    # split test and train
    train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test = combined[combined['is_train'] == 0].drop(columns=['is_train', 'is_fraud'], errors='ignore')

    return train, test, sample_submission


def run_xgboost(train, test, sample_submission):
    target = 'is_fraud'
    X = train.drop(columns=[target])
    y = train[target]
    X_test = test
    params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.01, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.85, 'min_child_weight': 2, 'gamma': 0.1, 'random_state': 42}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3500,  
            evals=[(dval, 'validation')],
            early_stopping_rounds=120,
        verbose_eval=None 
        )

        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / skf.n_splits

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Score: {oof_auc:.4f}")

    submission = sample_submission.copy()
    submission['is_fraud'] = (test_preds > 0.5).astype(int)
    submission.to_csv('submission.csv', index=False)
    print("Submission file: submission.csv")


train, test, sample_submission = load_and_preprocess_data()
run_xgboost(train, test, sample_submission)
