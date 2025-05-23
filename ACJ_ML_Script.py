# Загрузка необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Загрузка датасетов
train_df = pd.read_parquet('train_data.pqt')
test_df = pd.read_parquet('test_data.pqt')
cluster_weights_df = pd.read_excel("cluster_weights.xlsx")
sample_submission_df = pd.read_csv("sample_submission.csv")


# Определение функции потерь
def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)


# Коллонка с численным номером месяца
train_df['month_num'] = train_df['date'].apply(lambda x: x[-1]).astype(int)
test_df['month_num'] = test_df['date'].apply(lambda x: x[-1]).astype(int)

# Заполнение пропусков в month_6
m5_cluster = test_df[test_df['month_num'] == 5][['start_cluster']].reset_index()
m5_cluster['index'] += 1
m5_cluster.set_index('index', inplace=True)
m5_cluster = m5_cluster.to_dict('dict')['start_cluster']

m6_ids = test_df[test_df['month_num'] == 6].index
test_df.loc[m6_ids, 'start_cluster'] = m5_cluster

# Присвоение коллонке start_cluster типа категориальной переменной
all_clusters = train_df['start_cluster'].unique()
start_cluster_dtype = pd.CategoricalDtype(categories=all_clusters)

train_df['start_cluster'] = train_df['start_cluster'].astype(start_cluster_dtype)
test_df['start_cluster'] = test_df['start_cluster'].astype(start_cluster_dtype)

# Создание объединенного датасета для дальнейшей обработки
train_df['is_train'] = 1
test_df['is_train'] = 0
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values(['id', 'month_num']).reset_index(drop=True)

# Выделение остальных категориальных переменных

cat_cols = ["channel_code", "city", "city_type", "index_city_code", "ogrn_month",
            "ogrn_year", "okved", "segment", "start_cluster"]

for col in cat_cols:
    all_uniques = full_df[col].dropna().unique()
    cat_dtype = pd.CategoricalDtype(categories=all_uniques)
    full_df[col] = full_df[col].astype(cat_dtype)

# Добавление лаговых переменных

cols_for_lags = ['balance_amt_avg', 'balance_amt_max', 'balance_amt_min', 'balance_amt_day_avg',
                 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months']

for col in cols_for_lags:
    full_df[f'{col}_lag1'] = full_df.groupby('id')[col].shift(1).fillna(0)
    full_df[f'{col}_diff1'] = (full_df[col] - full_df[f'{col}_lag1']).fillna(0)

# Добавление категориальной переменной кластера в прошлом месяце
placeholder = "MISSING_LAG"
prev_month_clusters = list(all_clusters) + [placeholder]
prev_start_cluster_dtype = pd.CategoricalDtype(categories=prev_month_clusters)

full_df['prev_month_start_cluster'] = (full_df.groupby('id')['start_cluster'].shift(1)).astype(
    prev_start_cluster_dtype).fillna(placeholder)
cat_cols.append('prev_month_start_cluster')

# Обработка окончена, разделение датасетов по флагу is_train
train_processed_df = full_df[full_df['is_train'] == 1].copy()
test_processed_df = full_df[full_df['is_train'] == 0].copy()

# Перевод end_cluster в численных формат

# Датасеты соотношения названия и кода кластера
encoding_df = pd.DataFrame({'cluster': sorted(train_processed_df['end_cluster'].unique()),
                            'end_cluster_encoded': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
encoding_df_w_weights = encoding_df.merge(cluster_weights_df[['cluster', 'unnorm_weight']], on='cluster')

# Перенос кода в тренировочный датасет
train_processed_df = pd.merge(train_processed_df, encoding_df.rename({'cluster': 'end_cluster'}, axis=1),
                              on='end_cluster')

num_classes = len(encoding_df)

# Выделение переменных для тренировки
features_to_drop = ['id', 'date', 'month_num', 'is_train', 'end_cluster',
                    'end_cluster_encoded']  # Переменные, не учитываемы в обучении модели

features = [col for col in train_processed_df.columns if col not in features_to_drop]
final_cat_features = [col for col in cat_cols]

# Подготовка данных для обучения
X = train_processed_df[features]
y = train_processed_df['end_cluster_encoded']
X_test = test_processed_df[test_processed_df['month_num'] == 6][features].copy()

oof_preds = np.zeros((X.shape[0], num_classes))
test_preds = np.zeros((X_test.shape[0], num_classes))

# Обучение модели с использованием ассемблеи
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0], num_classes))
test_preds = np.zeros((X_test.shape[0], num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective='multiclass', metric='multi_logloss', num_class=num_classes,
        n_estimators=1000, learning_rate=0.05, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1, seed=42 + fold, boosting_type='gbdt'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=True)],
              categorical_feature=final_cat_features)

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / skf.n_splits

submission_df = pd.DataFrame(test_preds, columns=sorted(all_clusters))
submission_df['id'] = test_processed_df[test_processed_df['month_num'] == 6]['id'].values

submission_cols_order = ['id'] + list(sample_submission_df.columns[1:])
submission_df = submission_df[submission_cols_order]

submission_df.to_csv("submission.csv", index=False)
print("Submission file succesfully created!")
