import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import argparse
import mlflow
import mlflow.xgboost
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path, is_hybrid):
    """
    Loads and prepares the dataset by reading from a CSV file,
    dropping rows with missing values in specific columns, and separating features and target.

    Parameters:
    - file_path (str): The path to the CSV data file.
    - is_hybrid (bool): Whether the model is hybrid or not.

    Returns:
    - data (pd.DataFrame): The prepared dataset with features and target.
    """
    print("Loading and preparing data...")
    data = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}. Shape: {data.shape}")
    
    data = data.dropna(subset=['sea_water_temperature','estimatedStaticFOC'])
    print("Dropped rows with missing sea_water_temperature and estimatesStaticFOC")

    print("Data preparation complete. Returning data.")
    return data

def split_dataset_randomly(dataframe, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into training, validation, and test sets randomly.

    Parameters:
    - dataframe (pd.DataFrame): The complete dataset.
    - train_ratio (float): The proportion of the dataset to be used for training.
    - val_ratio (float): The proportion of the dataset to be used for validation.
    - test_ratio (float): The proportion of the dataset to be used for testing.

    Returns:
    - train_df (pd.DataFrame): The training dataset.
    - val_df (pd.DataFrame): The validation dataset.
    - test_df (pd.DataFrame): The test dataset.
    """
    print("Splitting dataset randomly...")
    np.random.seed(42)
    n = len(dataframe)
    test_size = int(n * test_ratio)
    test_indices = np.random.choice(dataframe.index, test_size, replace=False)
    test_df = dataframe.loc[test_indices]
    remaining_df = dataframe.drop(test_indices)
    n_remaining = len(remaining_df)
    train_size = int(n_remaining * train_ratio)
    val_size = int(n_remaining * val_ratio)
    train_df = remaining_df.iloc[:train_size]
    val_df = remaining_df.iloc[train_size:train_size + val_size]
    
    print(f"Random split complete. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
    return train_df, val_df, test_df

def split_dataset_sequentially(dataframe, train_ratio=0.7, val_ratio=0.2):
    """
    Splits the dataset into training, validation, and test sets sequentially.

    Parameters:
    - dataframe (pd.DataFrame): The complete dataset.
    - train_ratio (float): The proportion of the dataset to be used for training.
    - val_ratio (float): The proportion of the dataset to be used for validation.

    Returns:
    - train_df (pd.DataFrame): The training dataset.
    - val_df (pd.DataFrame): The validation dataset.
    - test_df (pd.DataFrame): The test dataset.
    """
    print("Splitting dataset sequentially...")
    n = len(dataframe)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = dataframe.iloc[:train_end]
    val_df = dataframe.iloc[train_end:val_end]
    test_df = dataframe.iloc[val_end:]
    
    print(f"Sequential split complete. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
    return train_df, val_df, test_df

def objective(trial, X_train_val, y_train_val):
    """
    Objective function for hyperparameter tuning using Optuna.

    Parameters:
    - trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
    - X_train_val (pd.DataFrame): Training and validation features.
    - y_train_val (pd.Series): Training and validation target.

    Returns:
    - mean_mse (float): The mean mean squared error of the model on the validation splits.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'objective': 'reg:squarederror',
    }
    model = XGBRegressor(**params)
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_index, test_index in tscv.split(X_train_val):
        X_cv_train, X_cv_test = X_train_val.iloc[train_index], X_train_val.iloc[test_index]
        y_cv_train, y_cv_test = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
        model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_test, y_cv_test)], verbose=False)
        preds = model.predict(X_cv_test)
        mse = mean_squared_error(y_cv_test, preds)
        mse_scores.append(mse)
    return np.mean(mse_scores)

def train_model(X_train_val, y_train_val, X_test, y_test, tune_model):
    """
    Trains the model using the given dataset and optionally tunes the hyperparameters using Optuna.

    Parameters:
    - X_train_val (pd.DataFrame): Combined training and validation features.
    - y_train_val (pd.Series): Combined training and validation target.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target.
    - tune_model (bool): Whether to perform hyperparameter tuning using Optuna.

    Returns:
    - y_pred (np.ndarray): The predicted values for the test set.
    - model (XGBRegressor): The trained XGBoost model.
    """
    if tune_model:
        print("Tuning model using Optuna...")
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train_val, y_train_val), n_trials=700)
        best_params = study.best_trial.params
        print("Hyperparameter tuning complete.")
    else:
        print("Using default parameters for model training.")
        best_params = {
            'n_estimators': 697,
            'learning_rate': 0.0104348354547033,
            'max_depth': 10,
            'subsample': 0.5193018469673057,
            'colsample_bytree': 0.5586316601020421,
            'lambda': 3.262741659046084
        }

    print("Training model...")
    model = XGBRegressor(**best_params)
    model.fit(X_train_val, y_train_val, verbose=False)
    y_pred = model.predict(X_test)
    print("Model training complete.")
    return y_pred, model

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model performance on the test set and prints various metrics.

    Parameters:
    - y_test (pd.Series): The actual target values for the test set.
    - y_pred (np.ndarray): The predicted values for the test set.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    print("Evaluating model performance...")
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred)
    mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"MAE (Mean Absolute Error) on Test Set: {mae_test:.2f}")
    print(f"MSE (Mean Squared Error) on Test Set: {mse_test:.2f}")
    print(f"RMSE (Root Mean Squared Error) on Test Set: {rmse_test:.2f}")
    print(f"R² (Coefficient of Determination) on Test Set: {r2_test:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error) on Test Set: {mape_test:.2f}%")
    print("Model evaluation complete.")

    return {
        'mae_test': mae_test,
        'mse_test': mse_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'mape_test': mape_test
    }


def split_and_evaluate_moving_window(data, features, target, no_est_params, train_window=8, val_window=1):
    """
    Splits the dataset using a moving window approach, trains the model, and evaluates performance.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing features and target.
    - features (list): List of feature column names.
    - target (str): Target column name.
    - no_est_params (dict): Parameters for the XGBoost model.
    - train_window (int): Number of months for the training period.
    - val_window (int): Number of months for the validation period.
    
    Returns:
    - results_df (pd.DataFrame): DataFrame containing evaluation metrics for each validation month.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data = data.sort_values(by='timestamp')

    data['year_month'] = data['timestamp'].dt.to_period('M')

    unique_months = data['year_month'].unique()

    results = []
    window_size = train_window + val_window

    for i in range(len(unique_months) - window_size + 1):
        train_start_month = unique_months[i]
        train_end_month = unique_months[i + train_window - 1]
        val_start_month = unique_months[i + train_window]
        val_end_month = unique_months[i + train_window + val_window - 1]

        train_data = data[(data['year_month'] >= train_start_month) & (data['year_month'] <= train_end_month)]
        val_data = data[(data['year_month'] == val_start_month)].copy()

        X_train = train_data[features]
        y_train = train_data[target]
        X_val = val_data[features]
        y_val = val_data[target]

        best_model = XGBRegressor(**no_est_params)
        best_model.fit(X_train, y_train, verbose=False)

        y_pred = best_model.predict(X_val)

        mae_corrected = mean_absolute_error(val_data['fuel_consumption'], y_pred)
        mse_corrected = mean_squared_error(val_data['fuel_consumption'], y_pred)
        rmse_corrected = np.sqrt(mse_corrected)
        r2_corrected = r2_score(val_data['fuel_consumption'], y_pred)
        mape_corrected = np.mean(np.abs((val_data['fuel_consumption'] - y_pred) / val_data['fuel_consumption'])) * 100

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

        results.append({
            'validation_month': val_start_month,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mae_corrected': mae_corrected,
            'mse_corrected': mse_corrected,
            'rmse_corrected': rmse_corrected,
            'r2_corrected': r2_corrected,
            'mape_corrected': mape_corrected
        })

    results_df = pd.DataFrame(results)
    return results_df

def plot_predictions_vs_actuals(speed_val, fuel_measurements, predictions, output_path):
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")

    sns.scatterplot(x=speed_val, y=fuel_measurements, color='orange', label='Measurements', s=50, alpha=0.4, edgecolor='k')
    sns.scatterplot(x=speed_val, y=predictions, color='blue', label='correction_model', s=50, alpha=0.4, edgecolor='none')

    plt.xlabel('Speed Over Ground', fontsize=14)
    plt.ylabel('Fuel Mass Flow', fontsize=14)
    plt.title('XGBOOST: FOC model with static estimates as a feature', fontsize=16)
    plt.legend(prop={'size': 12})

    plt.savefig(output_path)
    plt.close()

def plot_error_distribution(predictions, actuals, output_path):
    error_percentage = (predictions - actuals) / actuals * 100

    plt.figure(figsize=(12, 6))
    sns.histplot(error_percentage, bins=40, kde=False, color='skyblue')
    plt.yscale('log')
    plt.title('XGBOOST: FOC model with static estimates: Prediction Errors (%) with Logarithmic Scale')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency (Log Scale)')

    median_error = error_percentage.median()
    plt.axvline(x=median_error, color='r', linestyle='--', label=f'Median Error: {median_error:.2f}%')

    max_error = error_percentage.max()
    plt.annotate(f'Max Error: {max_error:.2f}%', xy=(max_error, 1), xytext=(max_error, 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right')

    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_real_vs_predicted(test_df, predictions, output_path1, output_path2):
    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, test_df['fuel_consumption'], 'g-', label='The real value', linewidth=2)
    plt.plot(test_df.index, predictions, 'r--', label='The predicted value', linewidth=2)
    plt.title('XGBOOST FOC model with estimates: Fuel Consumption: Real vs Predicted', fontsize=16)
    plt.xlabel('Data Point', fontsize=14)
    plt.ylabel('Fuel Consumption', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path1)
    plt.close()

    error = (predictions - test_df['fuel_consumption']) / test_df['fuel_consumption'] * 100

    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, error, 'b-', label='Error', linewidth=1)
    plt.title('XGBOOST FOC model with estimates: Prediction Error (%)', fontsize=16)
    plt.xlabel('Data Point', fontsize=14)
    plt.ylabel('Error Percentage', fontsize=14)
    plt.legend(fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path2)
    plt.close()

def plot_moving_window_results(results_df, window_size, output_path):
    """
    Plots the corrected model performance (MAPE) over time for the moving window approach.
    
    Parameters:
    - results_df (pd.DataFrame): DataFrame containing evaluation metrics for each validation month.
    - window_size (int): The total window size (train + validation) in months.
    - vessel_num (str): Identifier for the dataset (e.g., vessel number).
    - output_path (str): Path to save the generated plot.
    
    Returns:
    None
    """
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mape_corrected'])
    average_mape = results_df['mape_corrected'].mean()
    
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 9))
    
    threshold = 10
    colors = ['tab:blue' if value < threshold else 'tab:red' for value in results_df['mape_corrected']]
    
    bars = plt.bar(results_df['validation_month'].astype(str), results_df['mape_corrected'], color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), fontsize=10, ha='center', va='bottom', fontweight='bold', color='black')
    
    blue_patch = plt.Line2D([0], [0], color='tab:blue', lw=4, label=f'MAPE <= {threshold}%')
    red_patch = plt.Line2D([0], [0], color='tab:red', lw=4, label=f'MAPE > {threshold}%')
    plt.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=12)
    
    plt.xlabel('Validation Month', fontsize=14, fontweight='bold')
    plt.ylabel('MAPE Corrected (%)', fontsize=14, fontweight='bold', color='tab:blue')
    plt.title(f'Corrected Model Performance Over Time (MAPE), {window_size - 1} months training windows', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(results_df['mape_corrected']) + 5)
    plt.tight_layout()
    
    plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1)
    plt.text(len(results_df) - 1.5, threshold + 0.5, f'Threshold = {threshold}%', fontsize=12, color='gray', ha='right')
    
    plt.axhline(y=average_mape, color='green', linestyle='-', linewidth=1)
    plt.text(len(results_df) - 1.5, average_mape + 0.5, f'Average MAPE = {round(average_mape, 2)}%', fontsize=12, color='green', ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def main(file_path, tune_model, is_hybrid, use_random_split,use_moving_window):
    """
    Main function to load data, split datasets, train the model, and evaluate performance.

    Parameters:
    - file_path (str): The path to the CSV data file.
    - tune_model (bool): Whether to tune the model using Optuna.
    - is_hybrid (bool): Whether the model is hybrid or not.
    - use_random_split (bool): Whether to use random splitting of the dataset.

    Returns:
    None
    """
    print("Starting main workflow...")
    experiment_name = "XGBoost Hybrid FOC" if is_hybrid else "XGBoost FOC"
    mlflow.set_tracking_uri("http://128.214.253.154:5000/")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id[:8]  
        run_name = (
            f"Model: {experiment_name} | "
            f"Hybrid: {'Yes' if is_hybrid else 'No'} | "
            f"Tune: {'Yes' if tune_model else 'No'} | "
            f"Random Split: {'Yes' if use_random_split else 'No'} | "
            f"Moving Window: {'Yes' if use_moving_window else 'No'}"

        )
        
        mlflow.set_tag("mlflow.runName", run_name)
        
        print(f"Experiment: {experiment_name}")
        print(f"Run ID: {run_id}")
        print(f"Run Name: {run_name}")

        data = load_and_prepare_data(file_path, is_hybrid)
        
        if use_random_split:
            train_df, val_df, test_df = split_dataset_randomly(data)
        else:
            train_df, val_df, test_df = split_dataset_sequentially(data)

        columns_to_drop = ['fuel_consumption','timestamp']
        if not is_hybrid and 'estimatedStaticFOC' in train_df.columns:
            columns_to_drop.append('estimatedStaticFOC')

        X_train = train_df.drop(columns=columns_to_drop)
        X_val = val_df.drop(columns=columns_to_drop)
        X_test = test_df.drop(columns=columns_to_drop)
        y_train = train_df['fuel_consumption']
        y_val = val_df['fuel_consumption']
        y_test = test_df['fuel_consumption']

        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        y_pred, model = train_model(X_train_val, y_train_val, X_test, y_test, tune_model)
        
        metrics = evaluate_model(y_test, y_pred)

        params_to_log = {
            'n_estimators': model.get_params()['n_estimators'],
            'learning_rate': model.get_params()['learning_rate'],
            'max_depth': model.get_params()['max_depth'],
            'subsample': model.get_params()['subsample'],
            'colsample_bytree': model.get_params()['colsample_bytree'],
            'lambda': model.get_params()['lambda']
        }
        mlflow.log_params(params_to_log)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        plot_predictions_vs_actuals(X_test['speed_over_ground'], y_test, y_pred, "predictions_vs_actuals.png")
        mlflow.log_artifact("predictions_vs_actuals.png")

        plot_error_distribution(y_pred, y_test, "error_distribution.png")
        mlflow.log_artifact("error_distribution.png")

        plot_real_vs_predicted(test_df, y_pred, "real_vs_predicted.png", "prediction_error.png")
        mlflow.log_artifact("real_vs_predicted.png")
        mlflow.log_artifact("prediction_error.png")

        plot_importance(model)
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")


        print("Evaluating the model with monthly splits...")
        if use_moving_window:
            features = X_train_val.columns.tolist()
            target = 'fuel_consumption'
            no_est_params = model.get_params()
            results_df = split_and_evaluate_moving_window(data, features, target, no_est_params)
            print(results_df)
            plot_moving_window_results(results_df, window_size=8+1, output_path='monthly_eval.png')
            mlflow.log_artifact("monthly_eval.png")
        print("Workflow complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuel Consumption Model Training')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--tune_model', type=str, default='False', help='Whether to tune the model using Optuna')
    parser.add_argument('--is_hybrid', type=str, default='False', help='Whether the model is hybrid')
    parser.add_argument('--use_random_split', type=str, default='True', help='Whether to use random splitting of the dataset')
    parser.add_argument('--use_moving_window', type=str, default='False', help='Whether to use moving window splitting for evaluation')

    args = parser.parse_args()
    main(args.file_path, args.tune_model.lower() in ['true', '1', 'yes'], args.is_hybrid.lower() in ['true', '1', 'yes'], args.use_random_split.lower() in ['true', '1', 'yes'], args.use_moving_window.lower() in ['true', '1', 'yes'])



# To run this python file: 

#python fuel_consumption_XGBoost_model.py --file_path '../data/steady_state_with_stamps1.csv' --tune_model False --is_hybrid True --use_random_split False --use_moving_window True