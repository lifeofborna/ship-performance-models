import numpy as np
import torch
import torch.optim as optim
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import mlflow
import os
from datetime import datetime
import optuna 
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pytorch_lightning import seed_everything
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import argparse
import mlflow
from datetime import datetime

def reset_seeds(SEED):
    """
    Reset random seeds for reproducibility.

    Parameters:
    SEED (int): The seed value to use for random number generators.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    seed_everything(SEED, workers=True)

class dds(Dataset):
    """
    Custom PyTorch Dataset class for loading data from a DataFrame.

    Attributes:
    df (DataFrame): The DataFrame containing the data.

    Methods:
    __len__(): Returns the number of samples in the dataset.
    __getitem__(idx): Returns the feature tensor and target tensor for a given index.
    """

    def __init__(self, df, target):
        """
        Initialize the dataset by converting DataFrame columns to tensors.

        Parameters:
        df (DataFrame): The DataFrame containing the features and target values.
        """
        x = df.iloc[:, :-1].values
        y = df[target].values

        self.x_tensor = torch.tensor(x, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.y_tensor)

    def __getitem__(self, idx):
        """
        Return the feature and target tensor for a given index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: (feature tensor, target tensor) for the given index.
        """
        return self.x_tensor[idx], self.y_tensor[idx]

class fnn(torch.nn.Module):
    """
    Feedforward Neural Network (FNN) with customizable architecture.
    
    Args:
        input_dim (int): Number of input features.
        out_dim (int): Number of output features.
        hidden_layers (int): Number of hidden layers.
        hidden_sizes (list of int): Number of units in each hidden layer.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.
    """
    def __init__(self, input_dim, out_dim, hidden_layers, hidden_sizes, dropout_rate=0.1):
        super(fnn, self).__init__()
        
        layers = [
            torch.nn.Linear(input_dim, hidden_sizes[0]), 
            torch.nn.BatchNorm1d(hidden_sizes[0]), 
            torch.nn.LeakyReLU(0.2), 
            torch.nn.Dropout(dropout_rate)
        ]
        
        for i in range(1, hidden_layers):
            layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if i % 2 == 0:
                layers.append(torch.nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(torch.nn.LeakyReLU(0.2))
            layers.append(torch.nn.Dropout(dropout_rate))
        
        layers.append(torch.nn.Linear(hidden_sizes[-1], out_dim))
        
        self.layers = torch.nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, features):
        return self.layers(features)

class Trainer:
    """
    Trainer class for training and validating a PyTorch model with early stopping and optional learning rate scheduling.

    Attributes:
    model (nn.Module): The neural network model to be trained.
    criterion (nn.Module): The loss function.
    optimizer (optim.Optimizer): The optimizer.
    train_loader (DataLoader): DataLoader for training data.
    val_loader (DataLoader): DataLoader for validation data.
    n_epochs (int): Number of training epochs (default: 200).
    report_every (int): Frequency of reporting the training status (default: 1).
    patience (int): Number of epochs with no improvement after which training will be stopped (default: 100).
    scheduler (optional): Learning rate scheduler (default: None).
    valid_list (list): List to store validation losses.
    patience_counter (int): Counter for early stopping.
    best_val_loss (float): Best validation loss recorded.
    training_losses (list): List to store training losses.
    validation_losses (list): List to store validation losses.
    best_model_state (dict): State dictionary of the best model.
    
    Methods:
    train_one_epoch(): Train the model for one epoch.
    validate(): Validate the model.
    early_stopping(val_loss): Check and perform early stopping.
    train(): Train the model with early stopping.
    evaluate(test_loader): Evaluate the model on the test data.
    """

    def __init__(self, model, criterion, optimizer, train_loader, val_loader, n_epochs=200, report_every=1, patience=100, scheduler=None):
        """
        Initialize the Trainer with the model, criterion, optimizer, data loaders, and training parameters.

        Parameters:
        model (nn.Module): The neural network model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        n_epochs (int): Number of training epochs (default: 200).
        report_every (int): Frequency of reporting the training status (default: 1).
        patience (int): Number of epochs with no improvement after which training will be stopped (default: 100).
        scheduler (optional): Learning rate scheduler (default: None).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.patience = patience
        self.valid_list = []
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        self.best_model_state = None
        self.scheduler = scheduler

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
        float: The average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        for features, targets in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(features)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * features.size(0)
        return running_loss / len(self.train_loader.dataset)
    
    def validate(self):
        """
        Validate the model.

        Returns:
        float: The average validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feature_val, target_val in self.val_loader:
                prediction_val = self.model(feature_val)
                val_loss += self.criterion(prediction_val, target_val).item() * feature_val.size(0)
        return val_loss / len(self.val_loader.dataset)

    def early_stopping(self, val_loss):
        """
        Check and perform early stopping if there is no improvement in validation loss.

        Parameters:
        val_loss (float): Current validation loss.

        Returns:
        bool: True if early stopping is triggered, otherwise False.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict()
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print("Early stopping triggered.")
            return True
        return False

    def train(self):
        """
        Train the model with early stopping and optional learning rate scheduling.

        Returns:
        tuple: Best model state and final validation loss.
        """
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)

            if self.scheduler:
                self.scheduler.step(val_loss)
                
            if (epoch + 1) % self.report_every == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs}: Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

            if self.early_stopping(val_loss):
                print(f"Stopped early at epoch {epoch+1}.")
                break

        return (self.best_model_state, val_loss) if self.best_model_state is not None else (self.model.state_dict(), val_loss)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test data.

        Parameters:
        test_loader (DataLoader): DataLoader for test data.

        Returns:
        float: Average test loss.
        """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        total_test_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for features, targets in test_loader:
                predictions = self.model(features)
                batch_loss = self.criterion(predictions, targets)
                total_test_loss += batch_loss.item() * features.size(0)
                total_samples += features.size(0)
        
        average_test_loss = total_test_loss / total_samples

        return average_test_loss

def normalize_datasets(train_df, val_df, test_df, is_hybrid=False, target='fuel_consumption'):
    """
    Normalizes the datasets using a StandardScaler.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
        test_df (pd.DataFrame): Test dataset.
        use_hybrid (bool): Whether to include the hybrid feature 'estimatedStaticFOC'.
    
    Returns:
        tuple: A tuple containing three pandas DataFrames (normalized train_df, val_df, test_df)
               and the scaler used for normalization.
    """
    scaler = StandardScaler()
    
    features = [
        'speed_over_ground', 'wind_speed', 'swellheight', 'swellperiod', 'water_depth_combined', 
        'sea_water_temperature', 'current_speed', 'mean_draft', 'trim', 
        'relative_wind_direction_cog', 'relative_wave_direction_cog',
        'relative_swell_direction_cog', 'relative_current_direction_cog'
    ]
    
    if is_hybrid:
        features.append('estimatedStaticFOC')
    
    train_df_normalized = train_df.copy()
    val_df_normalized = val_df.copy()
    test_df_normalized = test_df.copy() if test_df is not None else None
    
    train_df_normalized[features] = scaler.fit_transform(train_df[features])
    val_df_normalized[features] = scaler.transform(val_df[features])
    if test_df_normalized is not None:
        test_df_normalized[features] = scaler.transform(test_df[features])
    
    return train_df_normalized, val_df_normalized, test_df_normalized, scaler

def load_and_prepare_data(filepath, monthly_eval=False, target='fuel_consumption'):
    """
    Load data from a CSV file, drop the 'timestamp' column if not performing monthly evaluation,
    remove rows with NaN values, and optionally calculate the error column.

    Args:
        filepath (str): The path to the CSV file.
        monthly_eval (bool, optional): Whether to retain the 'timestamp' column for monthly evaluation. Default is False.
        target (str, optional): The target column to predict. If 'error', an 'error' column will be calculated. Default is 'fuel_consumption'.

    Returns:
        pandas.DataFrame: The prepared data with the necessary columns and NaN values dropped.
    """

    data = pd.read_csv(filepath)
    
    if not monthly_eval:
        data = data.drop(columns=['timestamp'])    
    
    data = data.dropna()
    
    if target == 'error':
        if 'error' not in data.columns:
            data['error'] = data['fuel_consumption'] - data['estimatedStaticFOC']
    
    print(f"Columns after preparation: {data.columns}")
    return data

def split_data(data, train_ratio, val_ratio, target):
    """
    Split the data into training, validation, and test sets based on provided ratios.

    Args:
        data (pandas.DataFrame): The dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training.
        val_ratio (float): The ratio of the dataset to be used for validation.

    Returns:
        tuple: A tuple containing three pandas DataFrames: train_df, val_df, and test_df.
    """
    if target == 'error' and 'error' not in data.columns:
        data['error'] = data['fuel_consumption'] - data['estimatedStaticFOC']
    
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_df = data[:train_size]
    val_df = data[train_size:train_size + val_size]
    test_df = data[train_size + val_size:]
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df, batch_size, use_hybrid, target):
    """
    Create DataLoaders for the training, validation, and test datasets.

    Args:
        train_df (pandas.DataFrame): The training dataset.
        val_df (pandas.DataFrame): The validation dataset.
        test_df (pandas.DataFrame): The test dataset.
        batch_size (int): The batch size for the DataLoaders.
        use_hybrid (bool): Whether to use the hybrid model, which includes the 'estimatedStaticFOC' feature.
        target (str): The target column to predict. If 'error', the 'error' column will be used.

    Returns:
        tuple: A tuple containing three DataLoader objects: train_loader, val_loader, and test_loader (if test_df is not None).
    """
    if 'error' not in train_df.columns and target == 'error':
        train_df['error'] = train_df['fuel_consumption'] - train_df['estimatedStaticFOC']
        val_df['error'] = val_df['fuel_consumption'] - val_df['estimatedStaticFOC']
        test_df['error'] = test_df['fuel_consumption'] - test_df['estimatedStaticFOC']

    drop_cols = []
    if target != 'fuel_consumption':
        drop_cols.append('fuel_consumption')

    if target != 'error':
        drop_cols.append('error')

    if not use_hybrid:
        drop_cols.append('estimatedStaticFOC')

    train_df_features = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
    val_df_features = val_df.drop(columns=[col for col in drop_cols if col in val_df.columns])
    test_df_features = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])

    train_loader = DataLoader(dataset=dds(train_df_features, target), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dds(val_df_features, target), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=dds(test_df_features, target), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def setup_model_and_optimization(hyperparams, use_hybrid):
    """
    Set up the model, loss function, optimizer, and learning rate scheduler based on hyperparameters.

    Args:
        hyperparams (dict): A dictionary containing the hyperparameters for the model.
        use_hybrid (bool): A flag to determine the input dimension.

    Returns:
        tuple: A tuple containing the model, criterion, optimizer, and scheduler.
    """

    input_dim = 14 if use_hybrid else 13
    hidden_sizes = hyperparams['hidden_sizes']

    model = fnn(input_dim=input_dim, out_dim=1, hidden_layers=len(hidden_sizes), hidden_sizes=hidden_sizes, dropout_rate=hyperparams['dropout_rate'])
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    scheduler = None
    if hyperparams.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5, verbose=True
        )
    return model, criterion, optimizer, scheduler

def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs):
    """
    Train the model using the provided data loaders, criterion, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler.

    Returns:
        None
    """
    trainer = Trainer(
        model, criterion, optimizer, train_loader, val_loader, 
        n_epochs=epochs, report_every=10, patience=25, scheduler=scheduler
    )
    trainer.train()
    return trainer

def make_predictions(model, test_loader):
    """
    Make predictions using the trained model on the test dataset.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        pandas.Series: A series containing the predictions.
    """
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for features, _ in test_loader:
            if torch.isnan(features).any():
                print("NaN detected in input features")
                continue  
            predictions = model(features)
            if torch.isnan(predictions).any():
                print("NaN detected in predictions")
                continue 
            predictions_list.extend(predictions.cpu().numpy())
    return pd.Series([x[0] for x in predictions_list])

def denormalize_features(df, scaler, features):
    """
    Denormalize the specified features in the dataframe using the provided scaler.

    Args:
        df (pandas.DataFrame): The dataframe containing the normalized features.
        scaler (sklearn.preprocessing.StandardScaler or similar): The scaler used for normalization.
        features (list): List of feature names to be denormalized.

    Returns:
        None: The dataframe is modified in place.
    """
    original_features = scaler.inverse_transform(df[features])
    original_features_df = pd.DataFrame(original_features, columns=features, index=df.index)
    for feature in features:
        df[feature] = original_features_df[feature]

def evaluate_model(test_df, target):
    """
    Evaluate the model's performance using various metrics.

    Args:
        test_df (pandas.DataFrame): The dataframe containing the true values ('fuel_consumption') 
                                    and the predicted values ('Predictions').

    Returns:
        None: The function prints the evaluation metrics.
    """
    if target == 'error':
        y_true = test_df['fuel_consumption']
        y_pred = test_df['error_corrections']
    else:
        y_true = test_df[target]
        y_pred = test_df['predictions']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}")

    return mae, mse, rmse, r2, mape

def save_model(model, directory, filename):
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), model_path)

def objective_function(trial, use_hybrid_model, target):
    o_data = load_and_prepare_data('../data/steady_state_with_stamps1.csv', target=target)

    print("Splitting data into train, validation, and test sets for tuning...")
    train_df, val_df, test_df = split_data(o_data, train_ratio=0.7, val_ratio=0.2,target=target)
    normalize = True
    
    if normalize:
        print("Normalizing datasets for tuning...")
        train_df, val_df, test_df, scaler = normalize_datasets(train_df, val_df, test_df, is_hybrid=use_hybrid_model, target=target)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
    possible_hidden_sizes = [25, 50, 100, 150, 200, 250, 300]
    hidden_sizes = [trial.suggest_categorical(f"hidden_size_{i}", possible_hidden_sizes) for i in range(hidden_layers)]
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    possible_dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    dropout_rate = trial.suggest_categorical("dropout_rate", possible_dropout_rates)
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])

    best_hyperparams = {
        'hidden_layers': hidden_layers,
        'hidden_sizes': hidden_sizes,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'use_scheduler': use_scheduler
    }

    print(f"Hyperparameters Selected...: {best_hyperparams}")

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size, use_hybrid=use_hybrid_model, target=target)
    
    model, criterion, optimizer, scheduler = setup_model_and_optimization(best_hyperparams, use_hybrid=use_hybrid_model)

    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, n_epochs=100, report_every=10, patience=25, scheduler=scheduler)

    for epoch in range(100):
        train_loss = trainer.train_one_epoch()
        val_loss = trainer.validate()
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            print(f"Trial pruned at epoch {epoch + 1}")
            raise optuna.exceptions.TrialPruned()
        
        if trainer.early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return val_loss

def optimize_hyperparameters(n_trials, use_hybrid_model, target):
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3)
    )
    study.optimize(lambda trial: objective_function(trial, use_hybrid_model, target), 
                   n_trials=n_trials, n_jobs=-1, timeout=3600)  # 1 hour timeout

    best_trial = study.best_trial
    best_params = best_trial.params

    hidden_layers = best_params.pop('hidden_layers')
    hidden_sizes = [best_params.pop(f'hidden_size_{i}') for i in range(hidden_layers)]
    
    best_hyperparams = {
        'hidden_layers': hidden_layers,
        'hidden_sizes': hidden_sizes,
        'batch_size': best_params['batch_size'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'epochs': 100, 
        'dropout_rate': best_params['dropout_rate'],
        'use_scheduler': best_params['use_scheduler']
    }    

    print(f"Returning the tuned parameters... {best_hyperparams}")

    return best_hyperparams

def split_and_evaluate_moving_window(data, features, target, fnn_params, train_window=8, test_window=1, is_hybrid=False, max_runs=None):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')
    data['year_month'] = data['timestamp'].dt.to_period('M')
    unique_months = data['year_month'].unique()

    results = []
    window_size = train_window + test_window

    total_runs = len(unique_months) - window_size + 1
    if max_runs is not None:
        total_runs = min(total_runs, max_runs)
        
    if is_hybrid and 'estimatedStaticFOC' not in features:
        features.append('estimatedStaticFOC')
    
    print(f"\nStarting evaluation with {total_runs} runs:")
    print(f"Train window: {train_window} months, Test window: {test_window} months")
    print(f"Is hybrid model: {is_hybrid}")
    print(f"Target variable: {target}")
    print("=" * 50)

    for i in range(total_runs):
        train_start_month = unique_months[i]
        train_end_month = unique_months[i + train_window - 1]
        test_month = unique_months[i + train_window]

        print(f"\nRun {i+1}/{total_runs}:")
        print(f"Training period: {train_start_month} to {train_end_month}")
        print(f"Test period: {test_month}")

        train_data = data[(data['year_month'] >= train_start_month) & (data['year_month'] <= train_end_month)]
        test_data = data[data['year_month'] == test_month].copy()
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        if 'estimatedStaticFOC' not in test_data.columns:
            test_data['estimatedStaticFOC'] = data.loc[test_data.index, 'estimatedStaticFOC']
        
        model_features = features.copy()
        if not is_hybrid and 'estimatedStaticFOC' in model_features:
            model_features.remove('estimatedStaticFOC')
        
        train_data_model = train_data[model_features + [target]]
        test_data_model = test_data[model_features + [target]]

        train_data_model, test_data_model, _, scaler = normalize_datasets(train_data_model, test_data_model, None, is_hybrid=is_hybrid, target=target)

        train_size = int(0.8 * len(train_data_model))
        train_subset = train_data_model[:train_size]
        val_subset = train_data_model[train_size:]

        train_loader, val_loader, test_loader = create_dataloaders(train_subset, val_subset, test_data_model, fnn_params['batch_size'], is_hybrid, target)
        
        model, criterion, optimizer, scheduler = setup_model_and_optimization(fnn_params, use_hybrid=is_hybrid)

        trainer = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=fnn_params['epochs'])

        test_predictions = make_predictions(model, test_loader)

        denormalize_features(test_data_model, scaler, features=model_features)
        test_data_model['predictions'] = test_predictions.values

        test_data['predictions'] = test_data_model['predictions']

        if target == 'error':
            test_data['corrected_foc'] = test_data['estimatedStaticFOC'] + test_data['predictions']
        else:
            test_data['corrected_foc'] = test_data['predictions']

        mae = mean_absolute_error(test_data[target], test_data['corrected_foc'])
        mse = mean_squared_error(test_data[target], test_data['corrected_foc'])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_data[target], test_data['corrected_foc'])
        mape = np.mean(np.abs((test_data[target] - test_data['corrected_foc']) / test_data[target])) * 100

        print("\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.4f}%")

        results.append({
            'test_month': test_month,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        })

        remaining_runs = total_runs - i - 1
        print(f"\nCompleted run {i+1}/{total_runs}. Remaining runs: {remaining_runs}")
        print("-" * 50)

    results_df = pd.DataFrame(results)
    print("\nEvaluation complete. Returning results DataFrame.")
    return results_df

def expanding_window_evaluation(data, features, target, fnn_params, initial_window=1, is_hybrid=False, max_runs=None):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')
    data['year_month'] = data['timestamp'].dt.to_period('M')
    unique_months = data['year_month'].unique()

    results = []
    
    if is_hybrid and 'estimatedStaticFOC' not in features:
            features.append('estimatedStaticFOC')
    
    total_runs = len(unique_months) - initial_window
    if max_runs is not None:
        total_runs = min(total_runs, max_runs)

    print(f"\nStarting expanding window evaluation with {total_runs} runs:")
    print(f"Initial window: {initial_window} months")
    print(f"Is hybrid model: {is_hybrid}")
    print(f"Target variable: {target}")
    print("=" * 50)

    for i in range(initial_window, initial_window + total_runs):
        train_end_month = unique_months[i-1]
        test_month = unique_months[i]
        
        print(f"\nRun {i - initial_window + 1}/{total_runs}:")
        print(f"Training period: {unique_months[0]} to {train_end_month}")
        print(f"Test period: {test_month}")

        train_data = data[data['year_month'] <= train_end_month]
        test_data = data[data['year_month'] == test_month].copy()

        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        if 'estimatedStaticFOC' not in test_data.columns:
            test_data['estimatedStaticFOC'] = data.loc[test_data.index, 'estimatedStaticFOC']

        train_data = train_data[features + [target]]
        test_data_model = test_data[features + [target]]

        train_data, test_data_model, _, scaler = normalize_datasets(train_data, test_data_model, None, is_hybrid=is_hybrid, target=target)

        train_size = int(0.8 * len(train_data))
        train_subset = train_data[:train_size]
        val_subset = train_data[train_size:]

        train_loader, val_loader, test_loader = create_dataloaders(train_subset, val_subset, test_data_model, fnn_params['batch_size'], is_hybrid, target)

        model, criterion, optimizer, scheduler = setup_model_and_optimization(fnn_params, use_hybrid=is_hybrid)

        trainer = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=fnn_params['epochs'])

        test_predictions = make_predictions(model, test_loader)

        denormalize_features(test_data_model, scaler, features=features)
        test_data_model['predictions'] = test_predictions.values

        test_data['predictions'] = test_data_model['predictions']

        if target == 'error':
            test_data['corrected_foc'] = test_data['estimatedStaticFOC'] + test_data['predictions']
        else:
            test_data['corrected_foc'] = test_data['predictions']

        mae = mean_absolute_error(test_data[target], test_data['corrected_foc'])
        mse = mean_squared_error(test_data[target], test_data['corrected_foc'])
        rmse = np.sqrt(mse)
        r2 = r2_score(test_data[target], test_data['corrected_foc'])
        mape = np.mean(np.abs((test_data[target] - test_data['corrected_foc']) / test_data[target])) * 100

        print("\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.4f}%")

        results.append({
            'train_months': i,
            'test_month': test_month,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        })

        current_run = i - initial_window + 1
        remaining_runs = total_runs - current_run
        print(f"\nCompleted run {current_run}/{total_runs}. Remaining runs: {remaining_runs}")
        print("-" * 50)

    results_df = pd.DataFrame(results)
    print("\nEvaluation complete. Returning results DataFrame.")
    return results_df

def visualize_results(test_df, training_losses, validation_losses, target):
    plot_scatter(test_df, target)
    plot_error_histogram(test_df, target)
    plot_real_vs_predicted(test_df, target)
    plot_prediction_error(test_df, target)
    plot_training_valid_loss(training_losses, validation_losses)

def plot_scatter(test_df, target, save_path="plots/SOG_VS_FMF.jpg"):
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")
    sns.scatterplot(x=test_df['speed_over_ground'], y=test_df['fuel_consumption'], color='orange', label='Measurements', s=50, alpha=0.4, edgecolor='k')
    sns.scatterplot(x=test_df['speed_over_ground'], y=test_df['error_corrections' if target == 'error' else 'predictions'], color='blue', label='Predictions', s=50, alpha=0.4, edgecolor='none')
    plt.xlabel('Speed Over Ground', fontsize=14)
    plt.ylabel('Fuel Mass Flow' if target == 'fuel_consumption' else 'Error', fontsize=14)
    plt.title(f"Neural Network: {'Fuel Consumption' if target == 'fuel_consumption' else 'Error Correction'} Model", fontsize=16)
    plt.legend(prop={'size': 12})
    plt.savefig(save_path)
    plt.close()

def plot_error_histogram(test_df, target, save_path="plots/error_histogram.jpg"):
    error_percentage = (test_df['error_corrections' if target == 'error' else 'predictions'] - test_df['fuel_consumption']) / test_df['fuel_consumption'] * 100
    plt.figure(figsize=(12, 6))
    sns.histplot(error_percentage, bins=40, kde=False, color='skyblue')
    plt.yscale('log')
    plt.title('Prediction Errors (%) with Logarithmic Scale')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency (Log Scale)')
    median_error = error_percentage.median()
    plt.axvline(x=median_error, color='r', linestyle='--', label=f'Median Error: {median_error:.2f}%')
    max_error = error_percentage.max()
    plt.annotate(f'Max Error: {max_error:.2f}%', xy=(max_error, 1), xytext=(max_error, 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_real_vs_predicted(test_df, target, save_path="plots/real_vs_predicted.jpg"):
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, test_df['fuel_consumption'], 'g-', label='The Real Value', linewidth=2)
    plt.plot(test_df.index, test_df['error_corrections' if target == 'error' else 'predictions'], 'r--', label='The Predicted Value', linewidth=2)
    plt.title(f"{'Fuel Consumption' if target == 'fuel_consumption' else 'Error'}: Real vs Predicted", fontsize=16)
    plt.xlabel('Data Point', fontsize=14)
    plt.ylabel('Fuel Consumption' if target == 'fuel_consumption' else 'fuel_consumption', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_error(test_df, target, save_path="plots/prediction_error_percentage.jpg"):
    error = (test_df['error_corrections' if target == 'error' else 'predictions'] - test_df['fuel_consumption']) / test_df['fuel_consumption'] * 100
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, error, 'b-', label='Error', linewidth=1)
    plt.title('Prediction Error (%)', fontsize=16)
    plt.xlabel('Data Point', fontsize=14)
    plt.ylabel('Error Percentage', fontsize=14)
    plt.legend(fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_valid_loss(training_losses, validation_losses, save_path="plots/training_validation_loss.jpg"):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_moving_window_results(results_df, window_size, output_path):
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mape'])
    average_mape = results_df['mape'].mean()
    
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 9))
    
    threshold = 10
    colors = ['tab:blue' if value < threshold else 'tab:red' for value in results_df['mape']]
    
    bars = plt.bar(results_df['test_month'].astype(str), results_df['mape'], color=colors)
    
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
    plt.ylim(0, max(results_df['mape']) + 5)
    plt.tight_layout()
    
    plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1)
    plt.text(len(results_df) - 1.5, threshold + 0.5, f'Threshold = {threshold}%', fontsize=12, color='gray', ha='right')
    
    plt.axhline(y=average_mape, color='green', linestyle='-', linewidth=1)
    plt.text(len(results_df) - 1.5, average_mape + 0.5, f'Average MAPE = {round(average_mape, 2)}%', fontsize=12, color='green', ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
from scipy.signal import savgol_filter

from scipy.signal import savgol_filter
import numpy as np

def plot_expanding_window_results(results_df, output_path):
    results_df['validation_month'] = results_df['train_months'].astype(str)
    results_df['month_count'] = range(1, len(results_df) + 1)

    window_length = min(11, len(results_df) - 2) 
    window_length = window_length if window_length % 2 == 1 else window_length - 1  

    if len(results_df) >= 5:
        trend_line = savgol_filter(results_df['mape'], window_length=window_length, polyorder=min(2, window_length - 1))
    else:
        trend_line = results_df['mape'].rolling(window=min(3, len(results_df)), center=True).mean()

    plt.figure(figsize=(14, 8))
    plt.plot(results_df['month_count'], results_df['mape'], label='Corrected MAPE', color='tab:red', marker='o')
    plt.plot(results_df['month_count'], trend_line, label='Trend Line', color='tab:blue', linestyle='--', linewidth=2)
    
    for i, txt in enumerate(results_df['mape']):
        plt.annotate(f'{txt:.2f}', (results_df['month_count'][i], results_df['mape'][i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, fontweight='bold')
    
    plt.xlabel('Cumulative Month Count', fontsize=12, fontweight='bold')
    plt.ylabel('Corrected MAPE (%)', fontsize=12, fontweight='bold')
    plt.title('Corrected MAPE Over Increasing Training Data with Trend Line', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


def main(tune_model=True, is_hybrid=False, target='fuel_consumption', eval_type=None):
    """
    Main function to execute the workflow: reset seeds, optimize hyperparameters if necessary, load and prepare data,
    normalize datasets, create dataloaders, set up model and optimization, train the model, make predictions,
    denormalize features, evaluate the model, visualize results, and save the model.
    
    Args:
        tune_model (bool): Whether to perform hyperparameter tuning or use the existing architecture.
        is_hybrid (bool): Whether to use the "estimatedStaticFOC" feature for a hybrid model.
        target (str): The target variable, either 'fuel_consumption' or 'error'.
        eval_type (str): The type of evaluation to perform: 'sliding', 'expanding', or None for no evaluation.
    """

    experiment_name = f"FNN {'Hybrid ' if is_hybrid else ''}{'Error Correction' if target == 'error' else 'Fuel Consumption'} Model"
    if eval_type:
        experiment_name += f" - {eval_type.capitalize()} Window"
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id[:8]  

        run_name = f"{'Error Correction' if target == 'error' else 'Fuel Consumption'} Model {run_id} is_hybrid={is_hybrid}"
        if eval_type:
            run_name += f" eval_type={eval_type}"
        mlflow.set_tag("mlflow.runName", run_name)
    
        mlflow.log_param("tune_model", tune_model)
        mlflow.log_param("is_hybrid", is_hybrid)
        mlflow.log_param("target", target)
        mlflow.log_param("eval_type", eval_type)
        mlflow.log_param("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


        print("Resetting seeds...")
        reset_seeds(42)
        
        if tune_model:
            print("Optimizing hyperparameters...")
            best_hyperparams = optimize_hyperparameters(n_trials=1, use_hybrid_model=is_hybrid, target=target)
            mlflow.log_params(best_hyperparams)
            print("Best hyperparameters:", best_hyperparams)
        else:
            print("Using predefined hyperparameters...")
            best_hyperparams = {
                'epochs': 1,
                'learning_rate': 1.016172077518899e-05,
                'hidden_sizes': [177, 150, 125, 100],
                'hidden_layers': 4,
                'batch_size': 128,
                'dropout_rate': 0.43645610203085483,
                'use_scheduler': False,
                'weight_decay': 7.924516679174733e-05,
            }
            mlflow.log_params(best_hyperparams)

        print("Loading and preparing data...")
        o_data = load_and_prepare_data('../data/steady_state_with_stamps1.csv', target=target)

        print("Splitting data into train, validation, and test sets...")
        train_df, val_df, test_df = split_data(o_data, train_ratio=0.7, val_ratio=0.2, target=target)
        
        normalize = True
        
        if normalize:
            print("Normalizing datasets...")
        train_df, val_df, test_df, scaler = normalize_datasets(train_df, val_df, test_df, is_hybrid=is_hybrid, target=target)
        
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, best_hyperparams['batch_size'], use_hybrid=is_hybrid, target=target)
        
        print("Setting up model and optimization...")
        model, criterion, optimizer, scheduler = setup_model_and_optimization(best_hyperparams, use_hybrid=is_hybrid)
        
        print("--------------------------------")
        print('Selected Model Architecture: ', model)
        print("--------------------------------")

        print("Training model...")
        trainer = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=best_hyperparams['epochs'])
        
        print("Making predictions on test set...")
        predictions = make_predictions(model, test_loader)
        
        features_to_denormalize = [
            'speed_over_ground', 'wind_speed', 'swellheight', 'swellperiod', 'water_depth_combined', 
            'sea_water_temperature', 'current_speed', 'mean_draft', 'trim', 
            'relative_wind_direction_cog', 'relative_wave_direction_cog',
            'relative_swell_direction_cog', 'relative_current_direction_cog'
        ]
        
        if is_hybrid:
            features_to_denormalize.append('estimatedStaticFOC')
        
        print("Denormalizing features...")
        denormalize_features(test_df, scaler, features=features_to_denormalize)
        
        if target == 'error':
            test_df['error_corrections'] = test_df['estimatedStaticFOC'] + predictions.values
        else:
            test_df['predictions'] = predictions.values

        print("Evaluating model...")
        mae, mse, rmse, r2, mape = evaluate_model(test_df, target=target)
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)
        
        print("Visualizing results...")
        visualize_results(test_df, trainer.training_losses, trainer.validation_losses, target=target)

        mlflow.log_artifact("plots/training_validation_loss.jpg")
        mlflow.log_artifact("plots/error_histogram.jpg")
        mlflow.log_artifact("plots/prediction_error_percentage.jpg")
        mlflow.log_artifact("plots/real_vs_predicted.jpg")
        mlflow.log_artifact("plots/SOG_VS_FMF.jpg")

        run_id = run.info.run_id
        
        if eval_type:
            monthly_eval = load_and_prepare_data('../data/steady_state_with_stamps1.csv', monthly_eval=True, target=target)

            features = [
                'speed_over_ground', 'wind_speed', 'swellheight', 
                'swellperiod', 'water_depth_combined', 
                'sea_water_temperature', 'current_speed', 
                'mean_draft', 'trim', 
                'relative_wind_direction_cog', 'relative_wave_direction_cog',
                'relative_swell_direction_cog', 'relative_current_direction_cog'
            ]

            if eval_type == 'sliding':
                print("Performing sliding window evaluation...")
                results_df = split_and_evaluate_moving_window(
                    monthly_eval, 
                    features=features,
                    target=target, 
                    fnn_params=best_hyperparams, 
                    train_window=8, 
                    test_window=1, 
                    is_hybrid=is_hybrid,
                    max_runs=args.max_runs
                )

                plot_moving_window_results(results_df, window_size=8+1, output_path='monthly_eval.png')
                mlflow.log_artifact("monthly_eval.png")

            elif eval_type == 'expanding':
                print("Performing expanding window evaluation...")
                results_df = expanding_window_evaluation(
                    monthly_eval,
                    features=features,
                    target=target,
                    fnn_params=best_hyperparams,
                    initial_window=1,
                    is_hybrid=is_hybrid,
                    max_runs=args.max_runs
                )
                
                plot_expanding_window_results(results_df=results_df,output_path='expanding_windows.png')
                mlflow.log_artifact("expanding_windows.png")

            else:
                print("No window evaluation performed.")
                results_df = None

            if results_df is not None:
                results_df.to_csv(f'{eval_type}_window_results.csv', index=False)
                mlflow.log_artifact(f"{eval_type}_window_results.csv")

        print("Saving model...")
        model_path = f'models/{"error_correction" if target == "error" else "fuel_consumption"}_model{run_id}.pt'
        save_model(model, 'models', f'{"error_correction" if target == "error" else "fuel_consumption"}_model{run_id}.pt')
        
        mlflow.pytorch.log_model(model, "model")
        
        print("Workflow completed.")

        print(f"MLflow run completed with run_id: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuel Consumption Model Training')
    parser.add_argument('--tune_model', type=str, default='False', help='Whether to tune the model using Optuna')
    parser.add_argument('--use_hybrid_model', type=str, default='False', help='Whether to include estimatedStaticFOC')
    parser.add_argument('--target', type=str, default='fuel_consumption', choices=['fuel_consumption', 'error'], help='Target variable for prediction')
    parser.add_argument('--eval_type', type=str, default=None, choices=[None, 'sliding', 'expanding'], help='Type of window evaluation to perform')
    parser.add_argument('--max_runs', type=int, default=None, help='Maximum number of evaluation runs')
    
    args = parser.parse_args()
    main(args.tune_model.lower() in ['true', '1', 'yes'], 
         args.use_hybrid_model.lower() in ['true', '1', 'yes'],
         args.target,
         args.eval_type)
    

#python error_fnn_c.py --tune_model False --use_hybrid_model False --target error --eval_type expanding --max_runs 1
                                                                #  --target fuel_consumption  sliding