import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
from pytorch_lightning import seed_everything
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns

import mlflow
import os
from datetime import datetime

import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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

    def __init__(self, df):
        """
        Initialize the dataset by converting DataFrame columns to tensors.

        Parameters:
        df (DataFrame): The DataFrame containing the features and target values.
        """
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values  

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
    def __init__(self, input_dim, out_dim, hidden_layers, hidden_sizes, dropout_rate=0.1):
        super(fnn, self).__init__()
        layers = [
            torch.nn.Linear(input_dim, hidden_sizes), 
            torch.nn.BatchNorm1d(hidden_sizes), 
            torch.nn.LeakyReLU(0.2), 
            torch.nn.Dropout(dropout_rate)
        ]
        
        for i in range(1, hidden_layers):
            layers.append(torch.nn.Linear(hidden_sizes, hidden_sizes))
            if i % 2 == 0:
                layers.append(torch.nn.BatchNorm1d(hidden_sizes))
            layers.append(torch.nn.LeakyReLU(0.2))
            layers.append(torch.nn.Dropout(dropout_rate))
        
        layers += [
            torch.nn.Linear(hidden_sizes, out_dim), 
            torch.nn.ReLU()
        ]
        
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
            old_best_val_loss = self.best_val_loss
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
        previous_lr = None

        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)

            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                if previous_lr is None or current_lr != previous_lr:
                    print(f"Learning Rate changed after epoch {epoch+1}: {current_lr}")
                previous_lr = current_lr
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
    
def normalize_datasets_with_scaler(train_df, val_df, test_df=None,ishybrid=False):
    """
    Normalize the datasets using a standard scaler.

    Parameters:
    train_df (DataFrame): Training dataset.
    val_df (DataFrame): Validation dataset.
    test_df (DataFrame, optional): Test dataset (default: None).

    Returns:
    tuple: A dictionary containing normalized training, validation, and optional test datasets, 
           and the fitted StandardScaler object.
    """
    if ishybrid:
        features = train_df.columns.drop('fuel_consumption')
    else:
        features = train_df.columns.drop(['fuel_consumption', 'estimatedStaticFOC'])

    scaler = StandardScaler()
    
    train_df_normalized = train_df.copy()
    val_df_normalized = val_df.copy()
    
    train_df_normalized[features] = scaler.fit_transform(train_df[features])
    val_df_normalized[features] = scaler.transform(val_df[features])
    
    datasets = {
        'train': train_df_normalized,
        'val': val_df_normalized
    }
    
    if test_df is not None:
        test_df_normalized = test_df.copy()
        test_df_normalized[features] = scaler.transform(test_df[features])
        datasets['test'] = test_df_normalized
    
    return datasets, scaler



def objective_function(trial,use_hybrid_model):
    """
    Objective function for hyperparameter optimization using Optuna.

    Parameters:
    trial (optuna.trial.Trial): A Trial object to suggest hyperparameters.
    use_hybrid_model (bool): Whether to use the "estimatedStaticFOC" feature for a hybrid model.

    Returns:
    float: The validation loss of the model with the given hyperparameters.
    """

    o_data = load_and_prepare_data('../data/steady_state_with_stamps1.csv')

    print("Splitting data into train, validation, and test sets for tuning...")
    train_df, val_df, test_df = split_data(o_data, train_ratio=0.7, val_ratio=0.2)
    normalize = True
    
    if normalize:
        print("Normalizing datasets for tuning...")
        train_df, val_df, test_df, scaler = normalize_datasets(train_df, val_df, test_df,use_hybrid=use_hybrid_model)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_sizes = trial.suggest_int("hidden_size", 1, 300)
    hidden_layers =trial.suggest_int("hidden_layers", 1,20) 
    batch_size = trial.suggest_categorical("batch_size", [64,128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    best_hyperparams = {
        'hidden_layers': hidden_layers,
        'hidden_sizes': hidden_sizes,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    print(f"Hyperparameters Selected...: {best_hyperparams}")

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size, use_hybrid=use_hybrid_model)
    
    model, criterion, optimizer, scheduler = setup_model_and_optimization(best_hyperparams, use_hybrid=use_hybrid_model)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, n_epochs=100, report_every=10, patience=25,scheduler=scheduler)
    _,val_loss = trainer.train()
        
    return val_loss


def optimize_hyperparameters(n_trials, use_hybrid_model):
    """
    Optimize hyperparameters using Optuna.

    Args:
        n_trials (int): The number of trials for the Optuna study.
        use_hybrid_model (bool): Whether to use the "estimatedStaticFOC" feature for a hybrid model.

    Returns:
        dict: The best hyperparameters found by the Optuna study.
    """
    def objective(trial):
        return objective_function(trial, use_hybrid_model)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params

    hidden_layers = best_params.pop('hidden_layers')
    hidden_sizes =  best_params.pop('hidden_size')
    best_hyperparams = {
        'hidden_layers': hidden_layers,
        'hidden_sizes': hidden_sizes,
        'batch_size': best_params['batch_size'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'epochs': 100, 
        'dropout_rate': best_params['dropout_rate']
    }

    print(f"Returning the tuned parameters... {best_hyperparams}")

    return best_hyperparams

def load_and_prepare_data(filepath):
    """
    Load data from a CSV file, drop the 'timestamp' column, and remove rows with NaN values.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data with the 'timestamp' column removed and NaN values dropped.
    """
    data = pd.read_csv(filepath)
    data = data.drop(columns=['timestamp'])
    data = data.dropna()
    return data


def split_data(data, train_ratio, val_ratio):
    """
    Split the data into training, validation, and test sets based on provided ratios.

    Args:
        data (pandas.DataFrame): The dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training.
        val_ratio (float): The ratio of the dataset to be used for validation.

    Returns:
        tuple: A tuple containing three pandas DataFrames: train_df, val_df, and test_df.
    """
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size
    train_df = data[:train_size]
    val_df = data[train_size:train_size + val_size]
    test_df = data[train_size + val_size:]
    return train_df, val_df, test_df

def normalize_datasets(train_df, val_df, test_df,use_hybrid):
    """
    Normalize the training, validation, and test datasets using a scaler.

    Args:
        train_df (pandas.DataFrame): The training dataset.
        val_df (pandas.DataFrame): The validation dataset.
        test_df (pandas.DataFrame): The test dataset.

    Returns:
        tuple: A tuple containing three pandas DataFrames (normalized train_df, val_df, test_df) 
               and the scaler used for normalization.
    """
    datasets, scaler = normalize_datasets_with_scaler(train_df, val_df, test_df,use_hybrid)
    return datasets['train'], datasets['val'], datasets['test'], scaler


def create_dataloaders(train_df, val_df, test_df, batch_size, use_hybrid):
    """
    Create DataLoaders for the training, validation, and test datasets.

    Args:
        train_df (pandas.DataFrame): The training dataset.
        val_df (pandas.DataFrame): The validation dataset.
        test_df (pandas.DataFrame): The test dataset.
        batch_size (int): The batch size for the DataLoaders.
        use_hybrid (bool): Whether to use the hybrid model, which includes the 'estimatedStaticFOC' feature.

    Returns:
        tuple: A tuple containing three DataLoader objects: train_loader, val_loader, and test_loader.
    """
    if not use_hybrid:
        train_df_features = train_df.drop(columns=['estimatedStaticFOC'])
        val_df_features = val_df.drop(columns=['estimatedStaticFOC'])
        test_df_features = test_df.drop(columns=['estimatedStaticFOC'])
    else:
        train_df_features = train_df
        val_df_features = val_df
        test_df_features = test_df

    train_loader = DataLoader(dataset=dds(train_df_features), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=dds(val_df_features), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=dds(test_df_features), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def setup_model_and_optimization(hyperparams,use_hybrid):
    """
    Set up the model, loss function, optimizer, and learning rate scheduler based on hyperparameters.

    Args:
        hyperparams (dict): A dictionary containing the hyperparameters for the model.

    Returns:
        tuple: A tuple containing the model, criterion, optimizer, and scheduler.
    """

    input_dim = 14 if use_hybrid else 13

    hidden_sizes = hyperparams['hidden_sizes']

    model = fnn(input_dim=input_dim, out_dim=1, hidden_layers=hyperparams['hidden_layers'], hidden_sizes=hidden_sizes,dropout_rate=hyperparams['dropout_rate'])
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate']) # weight_decay=hyperparams['weight_decay']
   # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = None
    return model, criterion, optimizer, scheduler

def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler,epochs):
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

def evaluate_model(test_df):
    """
    Evaluate the model's performance using various metrics.

    Args:
        test_df (pandas.DataFrame): The dataframe containing the true values ('fuel_consumption') 
                                    and the predicted values ('Predictions').

    Returns:
        None: The function prints the evaluation metrics.
    """
    mae = mean_absolute_error(test_df['fuel_consumption'], test_df['Predictions'])
    mse = mean_squared_error(test_df['fuel_consumption'], test_df['Predictions'])
    rmse = mse ** 0.5
    r2 = r2_score(test_df['fuel_consumption'], test_df['Predictions'])
    mape = np.mean(np.abs((test_df['fuel_consumption'] - test_df['Predictions']) / test_df['fuel_consumption'])) * 100

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}")

    return mae,mse,rmse,r2,mape

def save_plot(plot_func, plot_path):
    plot_func()
    plt.savefig(plot_path)
    plt.close()

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def plot_scatter(test_df, save_path="plots/SOG_VS_FMF.jpg"):
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")
    sns.scatterplot(x=test_df['speed_over_ground'], y=test_df['fuel_consumption'], color='orange', label='Measurements', s=50, alpha=0.4, edgecolor='k')
    sns.scatterplot(x=test_df['speed_over_ground'], y=test_df['Predictions'], color='blue', label='Predictions', s=50, alpha=0.4, edgecolor='none')
    plt.xlabel('Speed Over Ground', fontsize=14)
    plt.ylabel('Fuel Mass Flow', fontsize=14)
    plt.title('Neural Network: Fuel Consumption Model with Static Estimates as a Feature', fontsize=16)
    plt.legend(prop={'size': 12})
    plt.savefig(save_path)
    plt.close()

def plot_error_histogram(test_df, save_path="plots/error_histogram.jpg"):
    error_percentage = (test_df['Predictions'] - test_df['fuel_consumption']) / test_df['fuel_consumption'] * 100
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

def plot_real_vs_predicted(test_df, save_path="plots/real_vs_predicted.jpg"):
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, test_df['fuel_consumption'], 'g-', label='The Real Value', linewidth=2)
    plt.plot(test_df.index, test_df['Predictions'], 'r--', label='The Predicted Value', linewidth=2)
    plt.title('Fuel Consumption: Real vs Predicted', fontsize=16)
    plt.xlabel('Data Point', fontsize=14)
    plt.ylabel('Fuel Consumption', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_error(test_df, save_path="plots/prediction_error_percentage.jpg"):
    error = (test_df['Predictions'] - test_df['fuel_consumption']) / test_df['fuel_consumption'] * 100
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

def visualize_results(test_df, training_losses, validation_losses):
    plot_scatter(test_df)
    plot_error_histogram(test_df)
    plot_real_vs_predicted(test_df)
    plot_prediction_error(test_df)
    plot_training_valid_loss(training_losses,validation_losses)


def save_model(model, directory, filename):
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), model_path)



    

def main(tune_model=True, use_hybrid_model=False):
    """
    Main function to execute the workflow: reset seeds, optimize hyperparameters if necessary, load and prepare data,
    normalize datasets, create dataloaders, set up model and optimization, train the model, make predictions,
    denormalize features, evaluate the model, visualize results, and save the model.
    
    Args:
        tune_model (bool): Whether to perform hyperparameter tuning or use the existing architecture.
        use_hybrid_model (bool): Whether to use the "estimatedStaticFOC" feature for a hybrid model.
    """

    experiment_name = "Neural Network Hybrid FOC Model" if use_hybrid_model else "Neural Network FOC Model"
    
    mlflow.set_tracking_uri("http://128.214.253.154:5000/")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id[:8]  


        run_name = f"Fuel Consumption Model {run_id} use_hybrid_model={use_hybrid_model}"
        mlflow.set_tag("mlflow.runName", run_name)
    
        mlflow.log_param("tune_model", tune_model)
        mlflow.log_param("use_hybrid_model", use_hybrid_model)
        mlflow.log_param("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        print("Resetting seeds...")
        reset_seeds(42)
        
        if tune_model:
            print("Optimizing hyperparameters...")
            best_hyperparams = optimize_hyperparameters(n_trials=1, use_hybrid_model=use_hybrid_model)
            mlflow.log_params(best_hyperparams)
            print("Best hyperparameters:", best_hyperparams)
        else:
            print("Using predefined hyperparameters...")
            best_hyperparams = {
                'hidden_layers': 1,
                'hidden_sizes': 126,
                'batch_size': 128,
                'learning_rate': 0.00220734692611745,
                'weight_decay': 7.924516679174733e-05,
                'epochs': 500,
                'dropout_rate': 0.1
            }
            mlflow.log_params(best_hyperparams)

        print("Loading and preparing data...")
        o_data = load_and_prepare_data('../data/steady_state_with_stamps1.csv')

        print("Splitting data into train, validation, and test sets...")
        train_df, val_df, test_df = split_data(o_data, train_ratio=0.7, val_ratio=0.2)
        normalize = True
        
        if normalize:
            print("Normalizing datasets...")
            train_df, val_df, test_df, scaler = normalize_datasets(train_df, val_df, test_df, use_hybrid=use_hybrid_model)
        
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, best_hyperparams['batch_size'], use_hybrid=use_hybrid_model)
        
        print("Setting up model and optimization...")
        model, criterion, optimizer, scheduler = setup_model_and_optimization(best_hyperparams, use_hybrid=use_hybrid_model)
        
        print("--------------------------------")
        print('Selected Model Architecture: ', model)
        print("--------------------------------")

        print("Training model...")
        trainer = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=best_hyperparams['epochs'])
        
        print("Making predictions on test set...")
        predictions = make_predictions(model, test_loader)

        test_df['Predictions'] = predictions.values
        
        features_to_denormalize = [
            'speed_over_ground', 'wind_speed', 'swellheight', 'swellperiod', 'water_depth_combined', 
            'sea_water_temperature', 'current_speed', 'mean_draft', 'trim', 
            'relative_wind_direction_cog', 'relative_wave_direction_cog',
            'relative_swell_direction_cog', 'relative_current_direction_cog'
        ]
        
        if use_hybrid_model:
            features_to_denormalize.append('estimatedStaticFOC')
        
        print("Denormalizing features...")
        denormalize_features(test_df, scaler, features=features_to_denormalize)
        
        print("Evaluating model...")
        mae, mse, rmse, r2, mape = evaluate_model(test_df)
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)
        
        print("Visualizing results...")
        visualize_results(test_df, trainer.training_losses, trainer.validation_losses)

        mlflow.log_artifact("plots/training_validation_loss.jpg")
        mlflow.log_artifact("plots/error_histogram.jpg")
        mlflow.log_artifact("plots/prediction_error_percentage.jpg")
        mlflow.log_artifact("plots/real_vs_predicted.jpg")
        mlflow.log_artifact("plots/SOG_VS_FMF.jpg")

        run_id = run.info.run_id

        print("Saving model...")
        model_path = f'models/fuel_consumption_model{run_id}.pt'
        save_model(model, 'models', f'fuel_consumption_model{run_id}.pt')
        
        mlflow.pytorch.log_model(model, "model")
        
        print("Workflow completed.")

        print(f"MLflow run completed with run_id: {run_id}")

if __name__ == "__main__":
    main(tune_model=True,use_hybrid_model=True)

#python fuel_consumption_model_FNN.py