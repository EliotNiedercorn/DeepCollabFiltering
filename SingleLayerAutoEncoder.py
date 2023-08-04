"""
Matrix completion using Denoising AutoEncoder with single hidden layer.

by Khalil Al Khouri, Eliot Niedercorn, Eliot Tomson and Alexis Misselyn the 28/05/2023 in Python 3.9
"""

import pandas as pd  # Used to read .xlsx file into pandas DataFrame and convert it to numpy matrix, might need to install "xlrd" with "pip install xlrd".
import numpy as np
import time
import matplotlib.pyplot as plt

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset


### 1. Parameters ######################################################################################################
device = "cpu"  # Use CPU.
num_epochs = 120  # Number of epochs to train for.
batch_size = 128  # Number of batch used during the training.
num_hidden_neurons = 12  # Size of the hidden layer.
learning_rate = 0.0001  # Learning rate use to update the values of the weights.
weight_decay = 0.001  # Weights-decay use in the regularization.
loss_fct = nn.MSELoss()  # Gives the square of the average deviation between the prediction and the actual value.

# Parameters of the Denoising AutoEncoder (cf. "Hybrid Recommender System based on Autoencoders")
alpha = 0.7  # Matrix denoising strength.
beta = 0.3  # Matrix reconstruction strength.
corruption_prob = 0.3  # Probability of corrupting during training.


### 2. Data Preparation ################################################################################################
xlsx2pandas = pd.read_excel('Data/jester_normalized.xlsx')  # Read the .xlsx file into a pandas DataFrame, switch to jester_normalized or movielens_normalized for the desired dataset.
matrix = xlsx2pandas.to_numpy()  # Convert the DataFrame to a numpy matrix.
print("Here is the matrix to complete, it's of shape:", matrix.shape)
print(matrix)

# Data splitting
train_validation_ratio = 0.9  # Here we took 90% training and 10% validation.
cutoff = int(matrix.shape[0] * train_validation_ratio)  # The cut between training and validation.
X_train = matrix[:cutoff, :]  # A matrix containing all rows before the cutoff and all the corresponding columns.
X_validation = matrix[cutoff:, :]  # A matrix containing all rows after the cutoff and all the corresponding columns.
X_train, X_validation = torch.from_numpy(X_train).to(device), torch.from_numpy(X_validation).to(device)  # Conversion to PyTorch GPU
train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)  # Create the training data loader.
validation_loader = torch.utils.data.DataLoader(X_validation, batch_size=batch_size, shuffle=True)  # Create the validation data loader.


### 3. AutoEncoder #####################################################################################################
class Autoencoder(nn.Module):  # Create a class that holds our neural network architecture.
    def __init__(self, input_neurons, hidden_neurons):
        super().__init__()
        self.encode = nn.Linear(input_neurons, hidden_neurons)  # Linear layer from the number of inputs neurons to the number of output neurons.
        self.decode = nn.Linear(hidden_neurons, input_neurons)  # Linear layer from the number of outputs neurons to the number of inputs neurons.

    def forward(self, values):
        encoded = F.relu(self.encode(values))  # Activation function used in the encoding layer.
        decoded = F.relu(self.decode(encoded))  # Activation function used in the decoding layer.
        return encoded, decoded


num_input_neurons = X_train.shape[1]  # Corresponds to the number of features in our train matrix.
autoencoder = Autoencoder(num_input_neurons, num_hidden_neurons).to(device)  # Instantation of our model.
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Declaration of our optimiser.


### 3. Training ########################################################################################################


def corrupt(tensor_to_corrupt):
    """
    Corrupts the given tensor by corrupting unmasked values at random.

    Args:
        tensor_to_corrupt (torch.Tensor): The input tensor to be corrupted.

    Returns:
        tuple: A tuple containing the following elements:
            - tensor_corrupted (torch.Tensor): A corrupted version of the input tensor with NaN values converted to 0.
            - total_corruption_mask (torch.Tensor): A mask for which the True elements corresponds to the corrupted values.
            - total_non_corruption_mask (torch.Tensor): A mask for which the True elements corresponds to the non corrupted values.
    """
    tensor_corrupted = tensor_to_corrupt.detach().clone()  # Create a cloned version of the tensor on which operations will be performed.
    NaNmask = torch.isnan(tensor_corrupted)  # Create a mask of the input tensor where the NaN values are boolean True.
    inverse_NaNmask = ~NaNmask  # Create the inverse mask.
    tensor_withoutNaN = tensor_corrupted[inverse_NaNmask]  # Create a version of the tensor with no NaN values on which corruption will be performed.
    corruption_mask = torch.rand(tensor_withoutNaN.shape) > corruption_prob  # Create a corruption mask with a probability of corruption_prob where True corresponds to keeping the value intact and the False corresponds to set the value to 0.
    tensor_corrupted[inverse_NaNmask] *= corruption_mask  # Apply the corruption mask on the version of the tensor without the NaN values.
    total_corruption_mask = torch.zeros_like(tensor_corrupted).bool()  # Instantiate a tensor on which information about the total corruption mask will be stored.
    total_corruption_mask[inverse_NaNmask] = ~corruption_mask  # Create the total corruption mask based on the corruption mask on the non NaN values.
    total_non_corruption_mask = ~total_corruption_mask * inverse_NaNmask  # Create the total non corruption mask by taking the inverse of the total corruption mask and applying the NaN mask.
    tensor_corrupted = torch.nan_to_num(tensor_corrupted, 0)  # Set all NaN values to 0
    return tensor_corrupted, total_corruption_mask, total_non_corruption_mask


start_time = time.time()  # Create a timer to count computational time.
train_losses = np.zeros(num_epochs)  # Create an array to store the losses during training.
validation_losses = np.zeros(num_epochs)  # Create an array to store the losses during validation.

for epoch in range(num_epochs):
    # Training
    running_loss = 0.0  # Variable that contains the losses for the given epoch.
    for batch in train_loader:
        model_output = batch.to(device).type(autoencoder.parameters().__next__().dtype)  # Model output for the current batch.
        output_corrupted, corruption_mask, non_corruption_mask = corrupt(model_output)  # Corruption of the model output and mask creation.
        model_output = torch.nan_to_num(model_output, 0)  # Set all NaN values to 0.
        _, model_output_corrupted = autoencoder(output_corrupted)  # Model output for the corrupted batch.
        loss = alpha * (loss_fct(model_output[corruption_mask], model_output_corrupted[corruption_mask])) + beta * (loss_fct(model_output[non_corruption_mask], model_output_corrupted[non_corruption_mask]))  # Computation of the loss (cf. "Hybrid Recommender System based on Autoencoders").
        loss.backward()  # Backpropagate the losses.
        running_loss += loss.item()  # Add the loss of the given batch.
        optimizer.step()  # Perfoms a single optimisation step.
        optimizer.zero_grad()  # Resets the gradient to 0, which is needed before calling it before the next batch.
    train_losses[epoch] = running_loss / len(train_loader)  # Stock the mean of the losses during the epoch into the train_losses array.
    elapsed_time = time.time() - start_time  # Calculate elapsed time.
    print(f"Epoch {epoch + 1}, Training Loss: {train_losses[epoch]}, Time passed: {elapsed_time} seconds")

    # Error takes into account the missing entries and thus the MSE is very low : only the reduction of error is important, scale is not

    # Validation
    running_loss = 0.0  # Variable that contains the losses for the given epoch.
    for data in validation_loader:
        with torch.no_grad():
            model_output_test = data.to(device).type(autoencoder.parameters().__next__().dtype)  # Model output for the current batch.
            output_corrupted_test, corruption_mask_test, non_corruption_mask_test = corrupt(model_output_test)  # Corruption of the model output and mask creation.
            model_output_test = torch.nan_to_num(model_output_test, 0)  # Set all NaN values to 0.
            _, model_output_corrupted_test = autoencoder(output_corrupted_test)   # Model output for the corrupted batch.
            loss = alpha * (loss_fct(model_output_test[corruption_mask_test], model_output_corrupted_test[corruption_mask_test])) + beta * (loss_fct(model_output_test[corruption_mask_test], model_output_corrupted_test[corruption_mask_test]))  # Computation of the loss (cf. "Hybrid Recommender System based on Autoencoders").
            running_loss += loss.item()  # Add the loss of the given batch.
    validation_losses[epoch] = running_loss / len(validation_loader)  # Stock the mean of the losses during the epoch into the validation_losses array.
    elapsed_time = time.time() - start_time  # Calculate elapsed time.
    print(f"Epoch {epoch + 1}, Validation Loss: {validation_losses[epoch]}, Time passed: {elapsed_time} seconds")


### 4. Plotting ########################################################################################################
fig, (s1, s2) = plt.subplots(1, 2)
s1.plot(train_losses, label='Training')
s2.plot(validation_losses, label='Validation', color='red')
plt.title('Results')
s1.set_ylabel('MSE')
s1.set_xlabel('Epochs')
s1.legend()
s2.legend()
plt.subplots_adjust(wspace=0.3)
plt.show()
