"""
This is a boilerplate pipeline 'cnn_regression'
generated using Kedro 0.19.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from math import sqrt
from typing import Dict
import pandas as pd
import numpy as np


# Data processing functions
def prepare_data(df: pd.DataFrame, parameters: Dict):
    FEATURE_COLUMNS = parameters['feature_columns']
    TARGET_COLUMN = parameters['target_column']
    GROUP_LABEL = parameters['group_label']

    features = torch.tensor(df[FEATURE_COLUMNS].values).float()
    targets = torch.tensor(df[TARGET_COLUMN].values).float().unsqueeze(1)
    groups = torch.tensor(df[GROUP_LABEL].values)

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(features.numpy(), groups=groups.numpy()))

    scaler = StandardScaler()
    train_features = features[train_idx]
    scaler.fit(train_features.numpy())

    scaled_train_features = torch.tensor(scaler.transform(train_features.numpy()))
    scaled_val_features = torch.tensor(scaler.transform(features[val_idx].numpy()))

    return scaled_train_features, targets[train_idx], scaled_val_features, targets[val_idx]

# Custom dataset class
class ResistanceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features.unsqueeze(2)
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

# Neural network model
class CNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Adaptive_CNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(out_channels, 32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Output of size (batch_size, 32, 1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_rmse(outputs, labels):
    mse = torch.mean((outputs - labels) ** 2)
    return torch.sqrt(mse)

def train_model(train_loader, val_loader, parameters: Dict):
    n_epochs = parameters["num_epochs"]
    input_channels = parameters["feature_columns"]
    out_channels = parameters["out_channels"]
    kernel_size = parameters["kernel_size"]
    model = CNNModel(in_channels=len(input_channels), out_channels=out_channels, kernel_size=kernel_size)
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    cnn_model = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        model.eval()
        val_rmse = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_rmse = calculate_rmse(outputs, labels) * inputs.size(0)
                val_rmse += batch_rmse.item()
                total_samples += inputs.size(0)

        val_rmse /= total_samples
        print(f'Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Val RMSE: {val_rmse:.4f}')


def wrap_loader(train_dataset, val_dataset, parameters: Dict):
    batch_size = parameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader

def create_ResistanceDataset(scaled_train_features, train_targets, scaled_val_features, val_targets):
    train_dataset = ResistanceDataset(scaled_train_features, train_targets)
    val_dataset = ResistanceDataset(scaled_val_features, val_targets)
    return train_dataset, val_dataset

# # Usage
# scaled_train_features, train_targets, scaled_val_features, val_targets = prepare_data(df)
# train_dataset = ResistanceDataset(scaled_train_features, train_targets)
# val_dataset = ResistanceDataset(scaled_val_features, val_targets)
# train_loader, val_loader = wrap_loader(train_dataset, val_dataset, {'batch_size': 32})
# train_model(train_loader, val_loader, n_epochs=15)
