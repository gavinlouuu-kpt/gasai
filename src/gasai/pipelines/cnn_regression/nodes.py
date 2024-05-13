"""
This is a boilerplate pipeline 'cnn_regression'
generated using Kedro 0.19.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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

def wrap_loader(train_dataset, val_dataset, parameters: Dict):
    batch_size = parameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader

def create_ResistanceDataset(scaled_train_features, train_targets, scaled_val_features, val_targets):
    train_dataset = ResistanceDataset(scaled_train_features, train_targets)
    val_dataset = ResistanceDataset(scaled_val_features, val_targets)
    return train_dataset, val_dataset

    
class CNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mid_channels, num_classes):
        super(CNNModel, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        
        # Second convolution layer - making mid_channels (intermediate output channels) configurable
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # First fully connected layer
        self.fc1 = nn.Linear(mid_channels, mid_channels // 2)  # dynamically setting size based on mid_channels
        
        # Second fully connected layer
        self.fc2 = nn.Linear(mid_channels // 2, num_classes)  # output size is now the number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze() # remove the extra dimension

def calculate_rmse(outputs, labels):
    mse = torch.mean((outputs - labels) ** 2)
    return torch.sqrt(mse)

def train_model(train_loader, val_loader, parameters: Dict):
    n_epochs = parameters["num_epochs"]
    input_channels = parameters["feature_columns"]
    num_classes = parameters["num_classes"]
    mid_channels = parameters["mid_channels"]
    out_channels = parameters["out_channels"]
    kernel_size = parameters["kernel_size"]
    learning_rate = parameters["learning_rate"]
    # print("This is the number of items in feature columns",len(input_channels))
    # Example instantiation
    model = CNNModel(in_channels=len(input_channels), out_channels=out_channels, kernel_size=kernel_size, mid_channels=mid_channels, num_classes=num_classes)
    # model = CNNModel(in_channels=len(input_channels), out_channels=out_channels, kernel_size=kernel_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

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
    # Save model
    return model

# Sliding window data preparation
def sequence_generate(data_set: pd.DataFrame, parameters: Dict):
    # Initialize lists to hold sequences and targets
    sequences = []
    targets = []
    features = parameters['feature_columns']
    target_column = parameters['target_column']
    sequence_length = parameters['sequence_length']
    step_size = parameters['step_size']

    # Group by 'exp_no' and create sequences for each group
    for _, group in data_set.groupby('exp_no'):
        data = group[features].values
        target_data = group[target_column].values
        # Create sequences with specified step size
        for i in range(0, len(group) - sequence_length, step_size):
            # Extract the sequence of features and the corresponding target
            sequence = data[i:(i + sequence_length)]
            target = target_data[i + sequence_length]  # Target is the next record
            sequences.append(sequence)
            targets.append(target)
    # Convert lists to numpy arrays
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    return sequences_np, targets_np

def split_data(sequences, targets, test_size=0.2, random_state=42):
    sequences_train, sequences_test, targets_train, targets_test = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state)
    return sequences_train, sequences_test, targets_train, targets_test


def scale_sequences(sequences_train, sequences_test):
    scaler = StandardScaler()
    n_samples_train, sequence_length, n_features = sequences_train.shape
    sequences_train_reshaped = sequences_train.reshape(-1, n_features)
    scaler.fit(sequences_train_reshaped)
    sequences_train_scaled = scaler.transform(sequences_train_reshaped).reshape(n_samples_train, sequence_length, n_features)
    
    n_samples_test, _, _ = sequences_test.shape
    sequences_test_reshaped = sequences_test.reshape(-1, n_features)
    sequences_test_scaled = scaler.transform(sequences_test_reshaped).reshape(n_samples_test, sequence_length, n_features)
    
    return sequences_train_scaled, sequences_test_scaled


def convert_to_tensors(sequences_train, sequences_test, targets_train, targets_test):
    train_sequences_tensor = torch.tensor(sequences_train, dtype=torch.float32).transpose(1,2)
    test_sequences_tensor = torch.tensor(sequences_test, dtype=torch.float32).transpose(1,2)
    train_targets_tensor = torch.tensor(targets_train, dtype=torch.float32)
    test_targets_tensor = torch.tensor(targets_test, dtype=torch.float32)
    # print("Shape on train sequence tensor", train_sequences_tensor.shape)
    return train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor


def create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor, test_targets_tensor, parameters: Dict):
    batch_size = parameters['batch_size']
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def sliding_data(data_set: pd.DataFrame, parameters: Dict):
    sequences_np, targets_np = sequence_generate(data_set, parameters)
    sequences_train, sequences_test, targets_train, targets_test = split_data(sequences_np, targets_np)
    sequences_train_scaled, sequences_test_scaled = scale_sequences(sequences_train, sequences_test)
    train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor = convert_to_tensors(
        sequences_train_scaled, sequences_test_scaled, targets_train, targets_test)
    train_loader, test_loader = create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor,
                                                   test_targets_tensor, parameters)
    return train_loader, test_loader
