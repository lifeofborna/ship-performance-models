import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import optuna
import os
from sklearn.model_selection import train_test_split

from pytorch_lightning import seed_everything


def reset_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    seed_everything(SEED, workers=True)


class dds(Dataset):
    def __init__(self, df):
        x = df.iloc[:, :14].values
        y = df.iloc[:, 14].values    
   
        self.x_tensor = torch.tensor(x, dtype = torch.float32)
        self.y_tensor = torch.tensor(y, dtype = torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]


def reset_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    seed_everything(SEED, workers=True)


class dds(Dataset):
    def __init__(self, df):
        x = df.iloc[:, :14].values
        y = df.iloc[:, 14].values    
   
        self.x_tensor = torch.tensor(x, dtype = torch.float32)
        self.y_tensor = torch.tensor(y, dtype = torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]


class fnn(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size):
        super(fnn, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc5 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc6 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc7 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc8 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc9 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc10 = torch.nn.Linear(hidden_size, out_dim)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        torch.nn.init.kaiming_normal_(self.fc7.weight)
        torch.nn.init.kaiming_normal_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.fc9.weight)
        torch.nn.init.kaiming_normal_(self.fc10.weight)

    def forward(self, features):
        output = self.lrelu(self.fc1(features))
        output = self.lrelu(self.fc2(output))
        output = self.lrelu(self.fc3(output))
        output = self.lrelu(self.fc4(output))
        output = self.lrelu(self.fc5(output))
        output = self.lrelu(self.fc6(output))
        output = self.lrelu(self.fc7(output))
        output = self.lrelu(self.fc8(output))
        output = self.lrelu(self.fc9(output))
        output = self.fc10(output)
        return output


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, n_epochs=200, report_every=1, patience=10):
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

    def train_one_epoch(self):
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
        if self.val_loader is None:
            print("Validation skipped because val_loader is None")
            return 0
        
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for feature_val, target_val in self.val_loader:
                prediction_val = self.model(feature_val)
                val_loss += self.criterion(prediction_val, target_val).item() * feature_val.size(0)
        return val_loss / len(self.val_loader.dataset)

    def early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).")
        else:
            self.patience_counter += 1
            print(f"Early Stopping Counter {self.patience_counter} of {self.patience}")

        if self.patience_counter >= self.patience:
            print("Early stopping triggered.")
            return True
        return False

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            if (epoch + 1) % self.report_every == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs}: Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

            if self.early_stopping(val_loss):
                print(f"Stopped early at epoch {epoch+1}.")
                break

        return self.best_val_loss

def normalize_datasets_with_scaler(train_df, val_df, test_df=None):
    features = train_df.columns.drop('fuelMassFlow')
    
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
    
    return datasets



def objective(trial):
    o_data = pd.read_csv('/home/borna/Desktop/work/EDA/sensor_data_container.csv')
    o_data = o_data.drop(columns=['timestamps'])

    train_size = int(len(o_data) * 0.7)
    val_size = int(len(o_data) * 0.2)
    
    train_df = o_data[:train_size]
    val_df = o_data[train_size:train_size + val_size]

    normalize = True
    if normalize: 
        datasets = normalize_datasets_with_scaler(train_df,val_df)
        train_df, val_df = datasets['train'],datasets['val']

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 10, 100)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    train_ds = dds(train_df)
    val_ds = dds(val_df)
    
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    
    model = fnn(input_dim=14, out_dim=1, hidden_size=hidden_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, n_epochs=100, report_every=10, patience=10)
    val_loss = trainer.train()
    
    return val_loss

if __name__ == "__main__":
    reset_seeds(42)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_hyperparams = study.best_trial.params
    print("Best hyperparameters:", best_hyperparams)
    
    o_data = pd.read_csv('/home/borna/Desktop/work/EDA/sensor_data_container.csv')  
    o_data = o_data.drop(columns=['timestamps'])

    train_size = int(len(o_data) * 0.7)
    val_size = int(len(o_data) * 0.2)
    test_size = len(o_data) - train_size - val_size  

    train_df = o_data[:train_size]
    val_df = o_data[train_size:train_size + val_size]
    test_df = o_data[train_size + val_size:] 
 
    normalize = True

    if normalize:
        datasets = normalize_datasets_with_scaler(train_df, val_df, test_df)
        train_df, val_df, test_df = datasets['train'], datasets['val'], datasets['test']

    train_val_ds = dds(train_df)
    train_loader = DataLoader(dataset=train_val_ds, batch_size=best_hyperparams['batch_size'], shuffle=False)
    
    valid_ds = dds(val_df)
    valid_loader = DataLoader(dataset=valid_ds,batch_size=best_hyperparams['batch_size'], shuffle=False)

    test_ds = dds(test_df)
    test_loader = DataLoader(dataset=test_ds, batch_size=best_hyperparams['batch_size'], shuffle=False)
    
    model = fnn(input_dim=14, out_dim=1, hidden_size=best_hyperparams['hidden_size'])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
    
    full_trainer = Trainer(model, criterion, optimizer, train_loader, valid_loader, n_epochs=100, report_every=10, patience=10)
    full_trainer.train()

    model.eval()
    total_test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in test_loader:
            predictions = model(features)
            batch_loss = criterion(predictions, targets)
            total_test_loss += batch_loss.item() * features.size(0) 
            total_samples += features.size(0)

    average_test_loss = total_test_loss / total_samples
    print(f"Average Test Loss: {average_test_loss}")

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'hyperparameter_nopreprocess.pt')
    torch.save(model.state_dict(), model_path)

