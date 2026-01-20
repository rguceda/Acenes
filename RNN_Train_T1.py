#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 18:49:07 2025

@author: Boris Pérez-Cañedo
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# Configuration
data_dir = 'Benceno T1'  # Update this path
input_size = 1  # Each sequence element is a scalar (Hammett constant)
hidden_size = 32
num_layers = 2
output_size = 1
batch_size = 32
num_epochs = 500
learning_rate = 0.001

train_split = 0.8  # 80% for training

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GRU Model
from RNN import GRURegressor

# Initialize dataset
from RNN import SequenceDataset, collate_fn
dataset = SequenceDataset(data_dir, "T1")

# Split dataset into training and testing
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model, loss, and optimizer
model = GRURegressor(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y, batch_lens in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        outputs = model(batch_x, batch_lens)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_lens in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x, batch_lens)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

print("Training complete!")
# Save the model
torch.save(model.state_dict(), 'T1_gru_regression_model.pth')
print("Model saved!")
