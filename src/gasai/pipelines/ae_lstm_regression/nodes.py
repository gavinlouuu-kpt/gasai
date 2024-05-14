import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict
import pandas as pd
import torch.optim as optim

# Autoencoder class definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input except for the batch dimension
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)
        
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Reshape back to the original sequence shape
        x = x.view(batch_size, seq_len, feature_dim)
        
        return x

def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    autoencoder.to(device)

    for epoch in range(num_epochs):
        autoencoder.train()
        total_train_loss = 0

        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            encoded = autoencoder(data)
            loss = criterion(encoded, data)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loss
        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                encoded = autoencoder(data)
                loss = criterion(encoded, data)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return autoencoder


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Data processing functions
def sequence_generate_and_split(data_set, parameters, test_size=0.2, random_state=42):
    features = parameters['feature_columns']
    target_column = parameters['target_column']
    sequence_length = parameters['sequence_length']
    step_size = parameters['step_size']

    sequences, targets, group_labels = [], [], []

    for exp_no, group in data_set.groupby('exp_no'):
        data = group[features].values
        target_data = group[target_column].values
        num_sequences = len(group) - sequence_length

        for i in range(0, num_sequences, step_size):
            sequence = data[i:(i + sequence_length)]
            target = target_data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
            group_labels.append(exp_no)

    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    group_labels_np = np.array(group_labels)

    unique_groups = np.unique(group_labels_np)
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)

    train_mask = np.isin(group_labels_np, train_groups)
    test_mask = np.isin(group_labels_np, test_groups)

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
    train_sequences_tensor = torch.tensor(sequences_train, dtype=torch.float32)
    test_sequences_tensor = torch.tensor(sequences_test, dtype=torch.float32)
    train_targets_tensor = torch.tensor(targets_train, dtype=torch.float32)
    test_targets_tensor = torch.tensor(targets_test, dtype=torch.float32)

    return train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor

def create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor, test_targets_tensor, parameters: Dict):
    batch_size = parameters['batch_size']
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    autoencoder.to(device)

    for epoch in range(num_epochs):
        autoencoder.train()
        total_train_loss = 0

        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            encoded, decoded = autoencoder(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loss
        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                encoded, decoded = autoencoder(data)
                loss = criterion(decoded, data)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return autoencoder

def extract_features(autoencoder, data_loader, device):
    autoencoder.to(device)
    autoencoder.eval()
    features = []

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device)  # Ensure data is a tensor and move to device
            encoded, _ = autoencoder(data)
            features.append(encoded.cpu().numpy())

    return np.concatenate(features)

def scale_and_encode_sequences(parameters: Dict, autoencoder, sequences_train, sequences_test, device):
    scaler = StandardScaler()
    sequences_train_reshaped = sequences_train.reshape(-1, sequences_train.shape[-1])
    sequences_test_reshaped = sequences_test.reshape(-1, sequences_test.shape[-1])

    scaler.fit(sequences_train_reshaped)
    sequences_train_scaled = scaler.transform(sequences_train_reshaped).reshape(sequences_train.shape)
    sequences_test_scaled = scaler.transform(sequences_test_reshaped).reshape(sequences_test.shape)

    train_dataset = TensorDataset(torch.tensor(sequences_train_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(sequences_test_scaled, dtype=torch.float32))

    batch_size = parameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_features = extract_features(autoencoder, train_loader, device)
    test_features = extract_features(autoencoder, test_loader, device)

    # Concatenate original data with extracted features
    train_combined = np.concatenate((sequences_train_scaled, train_features.reshape(-1, sequences_train.shape[1], train_features.shape[-1])), axis=2)
    test_combined = np.concatenate((sequences_test_scaled, test_features.reshape(-1, sequences_test.shape[1], test_features.shape[-1])), axis=2)

    return train_combined, test_combined

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_losses = []

    for sequences_batch, targets_batch in train_loader:
        sequences_batch = sequences_batch.to(device)
        targets_batch = targets_batch.to(device).unsqueeze(-1)

        outputs = model(sequences_batch)
        loss = criterion(outputs, targets_batch)
        epoch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_training_loss = sum(epoch_losses) / len(epoch_losses)
    return avg_training_loss

def validate(model, test_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    count = 0

    with torch.no_grad():
        for sequences_batch, targets_batch in test_loader:
            sequences_batch = sequences_batch.to(device)
            targets_batch = targets_batch.to(device).unsqueeze(-1)

            outputs = model(sequences_batch)
            loss = criterion(outputs, targets_batch)

            total_val_loss += loss.item()
            count += 1

    avg_val_loss = total_val_loss / count
    return avg_val_loss

def setup_training(model, learning_rate, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def train_model(train_loader, test_loader, parameters: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    features = parameters['features']
    input_size = parameters['sequence_length']
    hidden_size = parameters['hidden_size']
    num_layers = parameters['num_layers']
    num_classes = parameters['num_classes']

    model = RNN(input_size, hidden_size, num_layers, num_classes)
    learning_rate = parameters['learning_rate']

    criterion, optimizer = setup_training(model, learning_rate=learning_rate, device=device)

    num_epochs = parameters['num_epochs']
    training_losses = []
    validation_rmses = []

    for epoch in range(num_epochs):
        torch.backends.cudnn.enabled = True
        avg_training_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        training_losses.append(avg_training_loss)

        avg_val_loss = validate(model, test_loader, criterion, device)
        val_rmse = np.sqrt(avg_val_loss)
        validation_rmses.append(val_rmse)

        print(f'Epoch {epoch+1}, Training Loss: {avg_training_loss}, Validation RMSE: {val_rmse}')

    model.eval()
    return model

# Initialization and training functions
def ae_init_train(data_set, parameters: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ae_batch_size = parameters['ae_batch_size']
    sequences_train, sequences_test, targets_train, targets_test = sequence_generate_and_split(data_set, parameters)
    sequences_train_scaled, sequences_test_scaled = scale_sequences(sequences_train, sequences_test)

    train_dataset = TensorDataset(torch.tensor(sequences_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(sequences_test_scaled, dtype=torch.float32))

    ae_train_loader = DataLoader(train_dataset, batch_size=ae_batch_size, shuffle=True)
    ae_val_loader = DataLoader(val_dataset, batch_size=ae_batch_size, shuffle=False)


    input_dim = len(parameters['feature_columns'])
    encoding_dim = parameters['encoding_dim']
    ae_num_epochs = parameters['ae_num_epochs']
    ae_learning_rate = parameters['ae_learning_rate']
    autoencoder = Autoencoder(input_dim, encoding_dim)
    
    autoencoder = train_autoencoder(autoencoder, ae_train_loader, ae_val_loader, ae_num_epochs, ae_learning_rate, device)

    return autoencoder, sequences_train, sequences_test, targets_train, targets_test

def lstm_init_train(autoencoder, sequences_train, sequences_test, targets_train, targets_test, parameters: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_combined, test_combined = scale_and_encode_sequences(parameters, autoencoder, sequences_train, sequences_test, device)
    train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor = convert_to_tensors(train_combined, test_combined, targets_train, targets_test)
    train_loader, test_loader = create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor, test_targets_tensor, parameters)
    model = train_model(train_loader, test_loader, parameters)
    return model

def main(data_set, parameters: Dict):
    autoencoder, sequences_train, sequences_test, targets_train, targets_test = ae_init_train(data_set, parameters)
    model = lstm_init_train(autoencoder, sequences_train, sequences_test, targets_train, targets_test, parameters)
    return model

