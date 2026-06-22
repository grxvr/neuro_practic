import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna

# Налаштування
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 1e-6
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 1. Завантаження та обробка
def load_and_prepare(filepath='dataset.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не знайдено.")
    
    df = pd.read_csv(filepath, sep=';')
    df = df.rename(columns={'datetime': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
    df = df.sort_values('Datetime').drop_duplicates().reset_index(drop=True)
    
    # Ресемплінг часового ряду
    df = df.set_index('Datetime').resample('h').asfreq().reset_index()
    
    # Заповнення пропусків
    df['holiday'] = df['holiday'].fillna(0)
    weather_cols = [c for c in df.columns if 'om_best' in c]
    df[weather_cols] = df[weather_cols].interpolate(method='linear').ffill().bfill()
    
    return df

# 2. Генерація ознак
def create_features(df):
    df = df.copy()
    dt = df['Datetime'].dt
    
    # Часові ознаки
    df['hour'] = dt.hour
    df['dayofweek'] = dt.dayofweek
    df['month'] = dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(np.int32)
    
    # Циклічні ознаки
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Погодні взаємодії
    df['temp_cloud'] = df['om_best.168.apparent_temperature'] * df['om_best.168.cloudcover']
    df['temp_hum'] = df['om_best.168.apparent_temperature'] * df['om_best.168.relativehumidity_2m']
    
    # Створення лагів та ковзних вікон
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        df[f'target_lag_{lag}'] = df['target'].shift(lag)
        
    windows = [24, 48, 72]
    for w in windows:
        df[f'rolling_mean_{w}'] = df['target'].shift(1).rolling(window=w).mean()
        
    # Приведення до float32 для стабільності
    for col in df.columns:
        if col != 'Datetime':
            df[col] = df[col].astype(np.float32)
            
    return df

# 3. Рекурсивне прогнозування (оптимізоване)
def run_recursive_prediction(df, models, start_idx, end_idx, feature_cols):
    work_df = df.copy()
    # Використовуємо numpy array для швидких обчислень лагів
    target_vals = work_df['target'].values.astype(np.float32)
    
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    windows = [24, 48, 72]
    
    for i in range(start_idx, end_idx + 1):
        # Оновлення ознак (явне приведення до float32)
        for lag in lags:
            work_df.at[i, f'target_lag_{lag}'] = np.float32(target_vals[i-lag])
            
        for w in windows:
            work_df.at[i, f'rolling_mean_{w}'] = np.float32(np.mean(target_vals[i-w:i]))
            
        X_step = work_df.loc[[i], feature_cols]
        
        # Прогнози моделей
        p_lgb = models['LGBM'].predict(X_step)[0]
        p_xgb = models['XGB'].predict(X_step)[0]
        p_cat = models['CAT'].predict(X_step)[0]
        
        # Ансамбль (0.4/0.3/0.3)
        ensemble_pred = np.float32(0.4 * p_lgb + 0.3 * p_xgb + 0.3 * p_cat)
        
        # Оновлення значення
        target_vals[i] = ensemble_pred
        work_df.at[i, 'target'] = ensemble_pred
        
    return work_df.loc[start_idx:end_idx]

# 4. Оптимізація Optuna
def tune_lgbm(X, y):
    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'mape', 'verbosity': -1, 'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(calculate_mape(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

# --- Основна частина ---

print("Завантаження та генерація ознак...")
df_base = load_and_prepare('dataset.csv')
df_feats = create_features(df_base)

# Поділ даних (Holdout: останній тиждень з відомим target)
target_data = df_feats[df_feats['target'].notna()]
test_size = 168
train_data = target_data.iloc[:-test_size]
test_data = target_data.iloc[-test_size:]

feature_cols = [c for c in df_feats.columns if c not in ['Datetime', 'target']]
X_train, y_train = train_data[feature_cols], train_data['target']

print(f"Тренувальний період: {train_data['Datetime'].min()} - {train_data['Datetime'].max()}")

# Оптимізація
print("Оптимізація параметрів (LightGBM)...")
lgbm_params = tune_lgbm(X_train, y_train)

# Навчання ансамблю
print("Навчання ансамблю моделей...")
m_lgb = lgb.LGBMRegressor(**lgbm_params, n_estimators=1000, random_state=42, verbose=-1).fit(X_train, y_train)
m_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42).fit(X_train, y_train)
m_cat = CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=0).fit(X_train, y_train)

models = {'LGBM': m_lgb, 'XGB': m_xgb, 'CAT': m_cat}

# Рекурсивна валідація
print("Валідація на тестовому тижні (рекурсивно)...")
df_val = df_feats.copy()
df_val.loc[test_data.index, 'target'] = np.nan
val_results = run_recursive_prediction(df_val, models, test_data.index[0], test_data.index[-1], feature_cols)

# Метрики
mape = calculate_mape(test_data['target'], val_results['target'])
mae = mean_absolute_error(test_data['target'], val_results['target'])
rmse = np.sqrt(mean_squared_error(test_data['target'], val_results['target']))
r2 = r2_score(test_data['target'], val_results['target'])

print("\n" + "="*30)
print("МЕТРИКИ (HOLDOUT)")
print("="*30)
print(f"MAPE: {mape:.4f}%")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Фінальний прогноз
print("\nГенерація фінального прогнозу...")
# Рекурсія від останньої відомої точки (10.06) до кінця червня
forecast_start = target_data.index[-1] + 1
forecast_end = df_feats.index[-1]

df_final = df_feats.copy()
full_pred = run_recursive_prediction(df_final, models, forecast_start, forecast_end, feature_cols)
requested_output = full_pred[full_pred['Datetime'] >= '2026-06-25 03:00:00']

# Візуалізація
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

ax[0].plot(test_data['Datetime'], test_data['target'], label='Факт', color='black', alpha=0.7)
ax[0].plot(val_results['Datetime'], val_results['target'], label='Прогноз (рекурсія)', color='blue', linestyle='--')
ax[0].set_title(f'Валідація (MAPE: {mape:.2f}%)')
ax[0].legend()

ax[1].plot(requested_output['Datetime'], requested_output['target'], color='red', label='Прогноз')
ax[1].set_title('Прогноз на період 25.06.2026 - 30.06.2026')
ax[1].legend()

plt.tight_layout()
plt.savefig('prediction_plot.png', dpi=300)
plt.show()

# Збереження результатів
requested_output[['Datetime', 'target']].to_csv('forecast_results.csv', index=False, sep=';')
print("\nРезультати збережено у 'forecast_results.csv'. Графік збережено як 'prediction_plot.png'.")
