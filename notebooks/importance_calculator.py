# importance_calculator.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

def setup_directory(base_directory, session_timestamp):
    run_directory = os.path.join(base_directory, session_timestamp)
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)
    return run_directory

def compute_importances(model, sequence, device):
    ig = IntegratedGradients(model)
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    attributions, delta = ig.attribute(input_tensor, return_convergence_delta=True)
    return attributions.float().detach().cpu().numpy(), delta.detach().cpu().numpy()

def process_experiments(data_set, features, model, device, run_directory, sequence_length):
    
    for exp_no in data_set['exp_no'].unique():
        group = data_set[data_set['exp_no'] == exp_no]
        total_timesteps = len(group)
        number_of_features = len(features)
        cumulative_importances, contributions = calculate_importances(group, features, sequence_length, model, device)
        avg_importances = average_importances(cumulative_importances, contributions)
        plot_and_save_importances(avg_importances, exp_no, run_directory, total_timesteps, number_of_features)

def spec_exp_imp(data_set, features, model, device, run_directory, sequence_length, specific_exp_no, epoch_no):
    """
    Process experiments to compute feature importance only for a specific experimental number.

    Args:
    data_set (pd.DataFrame): The dataset containing the experimental data.
    features (list): List of features to calculate importances for.
    model (torch.nn.Module): The trained model.
    device (torch.device): The device to run the model computations on.
    run_directory (str): Directory where output plots will be saved.
    sequence_length (int): The length of the sequence used for model input.
    specific_exp_no (str, optional): The specific experimental number to compute importances for. Default is 'exp_0'.
    """

    # Check if the specific experimental number exists in the data
    if specific_exp_no in data_set['exp_no'].unique():
        group = data_set[data_set['exp_no'] == specific_exp_no]
        total_timesteps = len(group)
        number_of_features = len(features)

        # Calculate importances for the specified experiment
        cumulative_importances, contributions = calculate_importances(group, features, sequence_length, model, device)
        avg_importances = average_importances(cumulative_importances, contributions)

        # Plot and save the calculated importances
        plot_and_save_importances(avg_importances, specific_exp_no, epoch_no, run_directory, total_timesteps, number_of_features)
    else:
        print(f"No data available for experiment number {specific_exp_no}")


def calculate_importances(group, features, sequence_length, model, device):
    total_timesteps = len(group)
    number_of_features = len(features)
    cumulative_importances = np.zeros((total_timesteps, number_of_features))
    contributions = np.zeros(total_timesteps)

    for i in range(total_timesteps - sequence_length + 1):
        sequence = group[features].values[i:(i + sequence_length)]
        importances, _ = compute_importances(model, sequence, device)
        importances = importances.squeeze()  # Adjust squeezing here

        # Check if the shape is still not as expected, you can add more checks or reshape logic here
        if importances.ndim == 3:  # Assuming importances might sometimes be [1, sequence_length, num_features]
            importances = importances.squeeze(0)

        for idx in range(sequence_length):
            start_idx = i
            end_idx = start_idx + sequence_length
            imp_reshaped = importances[idx]  # Assuming importances shape is [sequence_length, num_features]
            cumulative_importances[start_idx:end_idx, :] += imp_reshaped
            contributions[start_idx:end_idx] += 1

    return cumulative_importances, contributions

def average_importances(cumulative_importances, contributions):
    avg_importances = np.zeros_like(cumulative_importances)
    non_zero_contributions = contributions > 0
    avg_importances[non_zero_contributions] = cumulative_importances[non_zero_contributions] / contributions[non_zero_contributions, None]
    return avg_importances

def plot_and_save_importances(avg_importances, exp_no, epoch_no, run_directory, total_timesteps, number_of_features):
    plt.figure(figsize=(10, 6))
    timesteps = np.arange(total_timesteps)
    for feature_idx in range(number_of_features):
        plt.plot(timesteps, avg_importances[:, feature_idx], label=f'Feature {feature_idx + 1}')
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Average Feature Importance')
    plt.title(f'Average Feature Importances Over Time for Exp {exp_no}')
    plt.savefig(os.path.join(run_directory, f'average_feature_importances_exp_{exp_no}_{epoch_no}.png'))
    plt.close()
