"""
This is a boilerplate pipeline 'transformer_regression'
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
        self.features = features  # Assuming features are already in the shape [batch_size, seq_length, feature_size]
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]



class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensure this is set to True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(feature_size, 1)  # Assuming regression to a single value

    def forward(self, src):
        transformer_output = self.transformer_encoder(src)
        # print("Transformer output shape:", transformer_output.shape)  # Debugging statement

        # Check if the sequence dimension is effectively used
        if transformer_output.dim() == 3 and transformer_output.size(1) > 1:  # Normal case with sequence
            out = transformer_output[:, -1, :]  # Using the last sequence element
        elif transformer_output.dim() == 2:  # Likely missing the sequence dimension
            out = transformer_output  # Directly use what's available
        else:
            raise ValueError("Unexpected transformer output shape: {}".format(transformer_output.shape))

        return self.fc_out(out)


def calculate_rmse(outputs, labels):
    mse = torch.mean((outputs - labels) ** 2)
    return torch.sqrt(mse)


def wrap_loader(train_dataset, val_dataset, parameters: Dict):
    batch_size = parameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader

def create_ResistanceDataset(scaled_train_features, train_targets, scaled_val_features, val_targets):
    train_dataset = ResistanceDataset(scaled_train_features, train_targets)
    val_dataset = ResistanceDataset(scaled_val_features, val_targets)
    return train_dataset, val_dataset


def train_model(train_loader, val_loader, parameters: Dict):
    # Check the shape of your input just before it is fed into the model
    # for inputs, labels in train_loader:
        # print("Batch input shape:", inputs.shape)  # Confirm the shape is [batch_size, sequence_length, feature_size]
        # outputs = model(inputs)


    n_epochs = parameters["num_epochs"]
    feature_size = len(parameters["feature_columns"])  # same as 'in_channels' in CNN
    num_layers = parameters["num_layers"]
    nhead = parameters["num_heads"]
    model = TransformerModel(feature_size=feature_size, num_layers=num_layers, nhead=nhead)
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

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

        # Evaluation and printing results similar to original code
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
        print(f'Epoch {epoch+1} Val RMSE: {val_rmse:.4f}')

# Continue with data preparation and other required changes similar to your existing setup
