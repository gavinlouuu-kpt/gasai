"""
This is a boilerplate pipeline 'transformer_regression'
generated using Kedro 0.19.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        self.features = features  # Assuming features are already in the shape [batch_size, seq_length, feature_size]
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]



# class TransformerModel(nn.Module):
#     def __init__(self, feature_size, num_layers, nhead, dim_feedforward=2048, dropout=0.1):
#         super(TransformerModel, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=feature_size,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True  # Ensure this is set to True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(feature_size, 1)  # Assuming regression to a single value

#     def forward(self, src):
#         transformer_output = self.transformer_encoder(src)
#         # print("Transformer output shape:", transformer_output.shape)  # Debugging statement

#         # Check if the sequence dimension is effectively used
#         if transformer_output.dim() == 3 and transformer_output.size(1) > 1:  # Normal case with sequence
#             out = transformer_output[:, -1, :]  # Using the last sequence element
#         elif transformer_output.dim() == 2:  # Likely missing the sequence dimension
#             out = transformer_output  # Directly use what's available
#         else:
#             raise ValueError("Unexpected transformer output shape: {}".format(transformer_output.shape))

#         return self.fc_out(out)



class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, num_outputs=1, dim_feedforward=2048, dropout=0.1, use_sequence=True):
        super(TransformerModel, self).__init__()
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_sequence = use_sequence  # Control whether to use last sequence element or whole sequence

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_size,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(self.feature_size, num_outputs)

    def forward(self, src):
        transformer_output = self.transformer_encoder(src)

        if self.use_sequence and transformer_output.dim() == 3 and transformer_output.size(1) > 1:
            out = transformer_output[:, -1, :]  # Use the last sequence element if sequence usage is enabled
        else:
            out = transformer_output.mean(dim=1)  # Use the mean of the sequence as the representation

        return self.fc_out(out).squeeze(-1)


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
    # feature_size = len(parameters["feature_columns"])  # same as 'in_channels' in CNN
    feature_size = parameters["sequence_length"]  # after generating sequence the feature size is the sequence length
    num_layers = parameters["num_layers"]
    nhead = parameters["num_heads"]
    num_classes = parameters["num_classes"]
    # model = TransformerModel(feature_size=feature_size, num_layers=num_layers, nhead=nhead)
    model = TransformerModel(
        feature_size=feature_size, 
        num_layers=num_layers, 
        nhead=nhead, 
        num_outputs=num_classes,  # Assume regression to a single value
        dim_feedforward=2048,  # Example, could be parameterized as well
        dropout=0.1  # Example, could be parameterized as well
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

#-----------
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

# Did not split according to group
def split_data(sequences, targets, test_size=0.2, random_state=42):
    sequences_train, sequences_test, targets_train, targets_test = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state)
    return sequences_train, sequences_test, targets_train, targets_test
#-----------

def sequence_generate_and_split(data_set, parameters, test_size=0.2, random_state=42):
    # Extract relevant parameters
    features = parameters['feature_columns']
    target_column = parameters['target_column']
    sequence_length = parameters['sequence_length']
    step_size = parameters['step_size']

    sequences, targets, group_labels = [], [], []

    # Generate sequences and targets for each group
    for exp_no, group in data_set.groupby('exp_no'):
        data = group[features].values
        target_data = group[target_column].values
        num_sequences = len(group) - sequence_length
        
        # Create sequences with the specified step size
        for i in range(0, num_sequences, step_size):
            sequence = data[i:(i + sequence_length)]
            target = target_data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
            group_labels.append(exp_no)

    # Convert lists to numpy arrays
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    group_labels_np = np.array(group_labels)

    # Split groups into training and testing groups
    unique_groups = np.unique(group_labels_np)
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)

    # Create masks for training and testing groups
    train_mask = np.isin(group_labels_np, train_groups)
    test_mask = np.isin(group_labels_np, test_groups)

    # Split sequences and targets using the masks
    sequences_train = sequences_np[train_mask]
    sequences_test = sequences_np[test_mask]
    targets_train = targets_np[train_mask]
    targets_test = targets_np[test_mask]

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
    # sequences_np, targets_np = sequence_generate(data_set, parameters)
    # sequences_train, sequences_test, targets_train, targets_test = split_data(sequences_np, targets_np)
    sequences_train, sequences_test, targets_train, targets_test = sequence_generate_and_split(data_set, parameters)
    sequences_train_scaled, sequences_test_scaled = scale_sequences(sequences_train, sequences_test)
    train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor = convert_to_tensors(
        sequences_train_scaled, sequences_test_scaled, targets_train, targets_test)
    train_loader, test_loader = create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor,
                                                   test_targets_tensor, parameters)
    return train_loader, test_loader
