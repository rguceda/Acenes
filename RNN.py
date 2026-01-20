#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:33:03 2025

@author: Boris Pérez-Cañedo
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset
import os
import pandas as pd


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        self.regressor = nn.Linear(hidden_size*self.num_directions, output_size)
        

    def forward(self, x, lengths):
        # Pack padded sequences
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.gru(packed_input)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Use the last non-padded element
        last_seq_idxs = lengths - 1
        last_outputs = output[torch.arange(output.size(0)), last_seq_idxs]
        regressor_output = self.regressor(last_outputs)
        return regressor_output

# Custom Dataset Class
class SequenceDataset(Dataset):
    def __init__(self, data_dir, target, sep=";", decimal="."):
        self.sequences = []
        self.targets = []
        self.lengths = []
        
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, file), sep=sep, decimal=decimal)
                pos_cols = [col for col in df.columns if col.startswith('Pos_')]
                for k, row in df.iterrows():
                    sequence = torch.FloatTensor(row[pos_cols].values.astype(float))
                    self.sequences.append(sequence)
                    self.targets.append(torch.FloatTensor([row[target]]))
                    self.lengths.append(len(pos_cols))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]

# Collate function for padding and sorting
def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)
    lengths = torch.LongTensor(lengths)
    
    # Sort by length (descending)
    lengths, sort_idx = lengths.sort(descending=True)
    sequences = [sequences[i] for i in sort_idx]
    targets = torch.stack([targets[i] for i in sort_idx])
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences.unsqueeze(-1), targets, lengths
