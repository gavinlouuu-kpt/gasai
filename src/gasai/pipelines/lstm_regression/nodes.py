
# from importance_calculator import process_experiments, setup_directory, spec_exp_imp
# load data from catalog
import os
import numpy as np
import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import time
import matplotlib.pyplot as plt
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients

# Fully connected neural network with one hidden layer
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         # Set initial hidden states (and cell states for LSTM)
#         device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
#         out, _ = self.lstm(x, (h0,c0))  
#         out = out[:, -1, :]
#         out = self.fc(out)
#         return out


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0, batch_norm_momentum=0.1):
#         super(RNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.batch_norm = nn.BatchNorm1d(hidden_size, momentum=batch_norm_momentum)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         # Set initial hidden states (and cell states for LSTM)
#         device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Apply batch normalization
#         out = out[:, -1, :]  # take the output of the last time step
#         out = self.batch_norm(out)
        
#         # Apply dropout
#         out = self.dropout(out)
        
#         out = self.fc(out)
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0, batch_norm_momentum=0.1, bidirectional=False):
        super(AttentionLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attention = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
        self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1), momentum=batch_norm_momentum)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        
    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Calculate attention weights
        attn_weights = torch.tanh(self.attention(out))
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Compute context vector as weighted sum of hidden states
        context_vector = torch.sum(attn_weights * out, dim=1)
        
        # Apply batch normalization
        context_vector = self.batch_norm(context_vector)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Fully connected layer
        out = self.fc(context_vector)
        return out

# Example usage:
# model = AttentionLSTM(input_size=10, hidden_size=20, num_layers=2, num_classes=1, dropout_prob=0.5, batch_norm_momentum=0.1, bidirectional=True)


# Example usage:
# model = IndependentLSTM(input_size=10, hidden_size=20, num_layers=2, num_classes=1, dropout_prob=0.5, batch_norm_momentum=0.1, bidirectional=True)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_training(model, learning_rate, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# def train_epoch(model, train_loader, criterion, optimizer, device):
#     model.train()
#     epoch_losses = []
#     for sequences_batch, targets_batch in train_loader:
#         sequences_batch = sequences_batch.to(device)
#         targets_batch = targets_batch.to(device).unsqueeze(-1)
        
#         outputs = model(sequences_batch)
#         loss = criterion(outputs, targets_batch)
#         epoch_losses.append(loss.item())
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     avg_training_loss = sum(epoch_losses) / len(epoch_losses)
#     return avg_training_loss

# reset states



# def validate(model, test_loader, criterion, device):
#     model.eval()
#     total_val_loss = 0
#     count = 0
#     with torch.no_grad():
#         for sequences_batch, targets_batch in test_loader:
#             sequences_batch = sequences_batch.to(device)
#             targets_batch = targets_batch.to(device).unsqueeze(-1)
            
#             outputs = model(sequences_batch)
#             loss = criterion(outputs, targets_batch)
            
#             total_val_loss += loss.item()
#             count += 1
    
#     avg_val_loss = total_val_loss / count
#     return avg_val_loss


# reset states
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Ensure the output and labels have the same shape
        outputs = outputs.squeeze()
        labels = labels.view(-1, 1).squeeze()

        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Ensure the output and labels have the same shape
            outputs = outputs.squeeze()
            labels = labels.view(-1, 1).squeeze()

            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(test_loader.dataset)
    return avg_loss



import torch
import matplotlib.pyplot as plt

def save_checkpoint(epoch, model, optimizer, avg_training_loss, val_rmse, session_checkpoint_dir):
    checkpoint_path = os.path.join(session_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': avg_training_loss,
        'validation_rmse': val_rmse,
    }, checkpoint_path)

def plot_metrics(training_losses, validation_rmses, epoch, session_checkpoint_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), validation_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE Over Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(session_checkpoint_dir, f'plots_epoch_{epoch+1}.png'))
    plt.close()


# Define the compute_importances function
def compute_importances(model, input_sequence):
    # Initialize IntegratedGradients with the model
    ig = IntegratedGradients(model)
    
    # Ensure the input sequence tensor is in float32 and add a batch dimension
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Compute attributions using Integrated Gradients
    attributions, delta = ig.attribute(input_tensor, return_convergence_delta=True)
    
    # Ensure the returned attributions are in float32
    attributions = attributions.float()
    
    # Return the attributions and delta as numpy arrays
    return attributions.detach().numpy(), delta.detach().numpy()

# Dictionary to store aggregated importances for each experiment
def layer_importances(model, data_set, features, sequence_length):
    experiment_importances = {}

    for exp_no, group in data_set.groupby('exp_no'):
        # Initialize a list to store importances for all sequences in this experiment
        all_importances = []

        # Compute importances for each sequence within this experiment
        for i in range(len(group) - sequence_length):
            sequence = group[features].values[i:(i + sequence_length)]
            # Assuming sequence is in the correct shape for the model
            importances = compute_importances(model, sequence)
            all_importances.append(importances)

        # Aggregate the importances across all sequences for this experiment
        # Here, we're taking the mean, but you could also sum them or use another method
        aggregated_importances = np.mean(all_importances, axis=0)

        # Store the aggregated importances in the dictionary
        experiment_importances[exp_no] = aggregated_importances

    return experiment_importances

def train_model(train_loader, test_loader, parameters: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    features = parameters['features'] 
    input_size = parameters['sequence_length']
    hidden_size = parameters['hidden_size']
    num_layers = parameters['num_layers']
    num_classes = parameters['num_classes']

    model = AttentionLSTM(input_size, hidden_size, num_layers, num_classes)
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
        val_rmse = math.sqrt(avg_val_loss)
        validation_rmses.append(val_rmse)

        print(f'Epoch {epoch+1}, Training Loss: {avg_training_loss}, Validation RMSE: {val_rmse}')


    # for epoch in range(num_epochs):
    #     torch.backends.cudnn.enabled = True
    #     avg_training_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    #     training_losses.append(avg_training_loss)

    #     avg_val_loss = validate(model, test_loader, criterion, device)
    #     val_rmse = math.sqrt(avg_val_loss)
    #     validation_rmses.append(val_rmse)

    #     print(f'Epoch {epoch+1}, Training Loss: {avg_training_loss}, Validation RMSE: {val_rmse}')

    # Save the model
    model.eval()
    return model


def sequence_generate_and_split(data_set, parameters, test_size=0.2, random_state=42):
    # Extract relevant parameters
    features = parameters['features']
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
    sequences_train, sequences_test, targets_train, targets_test = sequence_generate_and_split(data_set, parameters)
    sequences_train_scaled, sequences_test_scaled = scale_sequences(sequences_train, sequences_test)
    train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor = convert_to_tensors(
        sequences_train_scaled, sequences_test_scaled, targets_train, targets_test)
    train_loader, test_loader = create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor,
                                                   test_targets_tensor, parameters)
    return train_loader, test_loader
