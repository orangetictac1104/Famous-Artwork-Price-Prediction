import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# -------------------------------
# 1. Data loading and cleaning (unchanged)
# -------------------------------

DATA_DIR = "/Users/kelly/Downloads/artwork_details"
CSV_PATTERN = os.path.join(DATA_DIR, "*.csv")

def parse_currency(value: str) -> float:
    if pd.isna(value):
        return np.nan
    value = str(value).strip().replace('$', '').replace(',', '').strip()
    multiplier = 1.0
    if value.endswith('K'):
        multiplier = 1e3
        value = value[:-1]
    elif value.endswith('M'):
        multiplier = 1e6
        value = value[:-1]
    return float(value) * multiplier

def parse_multiplier(value: str) -> float:
    if pd.isna(value):
        return np.nan
    return float(str(value).strip().replace('x', ''))

def parse_years(value: str) -> int:
    if pd.isna(value):
        return np.nan
    match = re.search(r'(\d+)', str(value))
    return int(match.group(1)) if match else np.nan

def load_all_data():
    all_dfs = []
    for file_path in glob.glob(CSV_PATTERN):
        df = pd.read_csv(file_path)
        df = df[['artist', 'description', 'purchase_price', 'sale_price',
                 'gross_appreciation_multiplier', 'gross_appreciation_period']].copy()
        all_dfs.append(df)
    df_all = pd.concat(all_dfs, ignore_index=True)

    df_all['buy_price'] = df_all['purchase_price'].apply(parse_currency)
    df_all['sell_price'] = df_all['sale_price'].apply(parse_currency)
    df_all['appreciation_multiplier'] = df_all['gross_appreciation_multiplier'].apply(parse_multiplier)
    df_all['years_passed'] = df_all['gross_appreciation_period'].apply(parse_years)

    df_all = df_all.dropna(subset=['buy_price', 'sell_price', 'appreciation_multiplier', 'years_passed'])
    df_all['description'] = df_all['description'].fillna('unknown')

    return df_all[['artist', 'buy_price', 'years_passed', 'appreciation_multiplier',
                   'description', 'sell_price']].copy()

# -------------------------------
# 2. Log transform helpers
# -------------------------------

def log_transform(y):
    return np.log1p(y)

def inverse_log_transform(y_log):
    return np.expm1(y_log)

# -------------------------------
# 3. Training with progress tracking (OOB score)
# -------------------------------

def train_model_with_progress(df, n_estimators_final=100, step=10, random_state=42):
    """
    Train Random Forest incrementally, recording OOB R² after each 'step' trees.
    Also evaluates final model on test set.
    Returns (final_model, preprocessor, history)
    """
    categorical_features = ['artist', 'description']
    numeric_features = ['buy_price', 'years_passed', 'appreciation_multiplier']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    X = df[categorical_features + numeric_features]
    y = df['sell_price']
    y_log = log_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=random_state
    )

    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Initialize Random Forest with warm_start and oob_score
    rf = RandomForestRegressor(
        n_estimators=0,
        warm_start=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )

    history = {'n_estimators': [], 'oob_r2': []}
    current_trees = 0
    print(f"Training Random Forest incrementally (target final trees = {n_estimators_final})")
    print("=" * 60)

    while current_trees < n_estimators_final:
        rf.n_estimators += step
        rf.fit(X_train_processed, y_train)
        current_trees = rf.n_estimators
        oob_r2 = rf.oob_score_
        history['n_estimators'].append(current_trees)
        history['oob_r2'].append(oob_r2)
        print(f"Trees: {current_trees:3d} | OOB R²: {oob_r2:.6f}")

    print("=" * 60)

    # Final evaluation on test set
    y_pred_log = rf.predict(X_test_processed)
    y_pred = inverse_log_transform(y_pred_log)
    y_true = inverse_log_transform(y_test)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\nFinal Test Set Performance:")
    print(f"R² score: {r2:.4f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

    return rf, preprocessor, history

def plot_training_progress(history):
    """Plot OOB R² vs number of trees."""
    plt.figure(figsize=(8, 5))
    plt.plot(history['n_estimators'], history['oob_r2'], marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Number of Trees")
    plt.ylabel("OOB R² Score")
    plt.title("Random Forest Training Progress (OOB Score)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Save / load functions (unchanged)
# -------------------------------

def save_model(model, preprocessor, model_path='artwork_price_model.joblib',
               preprocessor_path='preprocessor.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\nModel saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

def load_model_and_preprocessor(model_path='artwork_price_model.joblib',
                                preprocessor_path='preprocessor.joblib'):
    return joblib.load(model_path), joblib.load(preprocessor_path)

def predict_sell_price(artist, buy_price, years_passed, appreciation_multiplier,
                       description, model, preprocessor):
    input_df = pd.DataFrame([{
        'artist': artist,
        'description': description,
        'buy_price': buy_price,
        'years_passed': years_passed,
        'appreciation_multiplier': appreciation_multiplier
    }])
    X_input = preprocessor.transform(input_df)
    log_pred = model.predict(X_input)[0]
    return inverse_log_transform(log_pred)

# -------------------------------
# 5. Main execution
# -------------------------------

if __name__ == "__main__":
    print("Loading and cleaning data...")
    data = load_all_data()
    print(f"Total rows after cleaning: {len(data)}")

    print("\nTraining Random Forest with progress logging...")
    model, preproc, history = train_model_with_progress(data, n_estimators_final=100, step=10)

    plot_training_progress(history)
    save_model(model, preproc)

    # Example prediction
    print("\n--- Example Prediction ---")
    pred_price = predict_sell_price(
        artist="ZAO WOU KI",
        buy_price=50000.0,
        years_passed=20,
        appreciation_multiplier=10.0,
        description="oil on canvas",
        model=model,
        preprocessor=preproc
    )
    print(f"Predicted sell price: ${pred_price:,.2f}")