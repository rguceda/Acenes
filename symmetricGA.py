#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 18:29:52 2026

@author: Boris Pérez-Cañedo
"""

import numpy as np
from deap import base, creator, tools, algorithms
from functools import partial


import torch
from RNN import GRURegressor

##################################

def closest_hammett(h, hammetts):
    closest = hammetts[0]
    distance = abs(h-closest)
    for hammett in hammetts[1:]:
        test_dist = abs(h-hammett)
        if test_dist <= distance:
            closest = hammett
            distance = test_dist
    return closest

def main(hammetts, symmetry_rules, use_continuous_hammett):
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # S1 and T1 RNN models
    input_size = 1
    hidden_size = 32
    num_layers = 2
    output_size = 1
    
    model_s1 = GRURegressor(input_size, hidden_size, num_layers, output_size).to(device)
    model_s1.load_state_dict(torch.load('S1_gru_regression_model.pth', weights_only=True))
    model_s1.eval()
    
    model_t1 = GRURegressor(input_size, hidden_size, num_layers, output_size).to(device)
    model_t1.load_state_dict(torch.load('T1_gru_regression_model.pth', weights_only=True))
    model_t1.eval()
    
    # DEAP
    for name in ["FitnessMin","Individual"]:
        if name in creator.__dict__:
            delattr(creator, name)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Acene length
    solution_length = 10 # 6 para Benceno, 8 para Naftaleno, 10 para Antraceno, 12 para Tetraceno, 14 para Pentaceno
    
    def create_individual(hammetts, symmetry_rules, prefered_hammetts=None):
        n_positions = solution_length
        ind = [0]*n_positions
        for i, rule in enumerate(symmetry_rules):
            if prefered_hammetts == None:
                non_zero_hammetts = [v for v in hammetts if v != 0]
                h = np.random.choice(non_zero_hammetts)
            else:
                h = prefered_hammetts[i]
            for pos in rule:
                ind[pos] = h
        return creator.Individual(ind)

    def crossover(ind1, ind2, hammetts, symmetry_rules, use_continuous_hammett):
        # Make copies of the individuals
        new1 = ind1[:]
        new2 = ind2[:]
        # Apply each symmetry rule individually
        for rule in symmetry_rules:
            # All the positions in a symmetry rule have the same Hammett constant
            # So, we can simply get the Hammett in the first position
            pos = rule[0] # The position
            hammett_ind1 = ind1[pos]
            hammett_ind2 = ind2[pos]
            avg_hammett = (hammett_ind1 + hammett_ind2)/2
            if use_continuous_hammett:
                h = avg_hammett
            else:
                # We stick to discrete Hammett constants
                # Find the closest Hammett in the original list of Hammett comnstants
                h = closest_hammett(avg_hammett, hammetts)
                # Don't use Hydrogen or the same Hammett constant of the parents (We could do something else)
                if h == 0 or h == hammett_ind1 or h == hammett_ind2:
                    # Get a different hammett constant
                    posibilities = [v for v in hammetts if v != 0 and v != hammett_ind1 and v != hammett_ind2]
                    h = np.random.choice(posibilities)
            for pos in rule:
                new1[pos] = h
                new2[pos] = h
        return creator.Individual(new1), creator.Individual(new2)
    
    def mutate(ind, hammetts, symmetry_rules, indpb=0.3):
        # Make a copy of the individual
        new = ind[:]
        for rule in symmetry_rules:
            # Decide whether to mutate this symmetry rule
            if np.random.rand() < indpb:
                # All the positions in a symmetry rule have the same Hammett constant
                # So, we can simply get the Hammett in the first position
                pos = rule[0]
                h = new[pos]
                possibilities = [v for v in hammetts if v != h and v != 0] # Avoid substituting the same Hammett constant or Hydrogen
                h = np.random.choice(possibilities)
                for pos in rule:
                    new[pos] = h
        return creator.Individual(new),
    
    def evaluate(ind):
        lengths = torch.tensor([len(ind)])
        input_sequence = torch.tensor(ind, dtype=torch.float).view(1, len(ind), 1)
        S1 = float(model_s1(input_sequence, lengths))
        T1 = float(model_t1(input_sequence, lengths))
        
        # Penalty if S1 < 2*T1
        penalty = max(0, 2*T1 - S1)
        return (penalty,)

    # Put our operators inside the Toolbox

    toolbox.register("individual", tools.initIterate, creator.Individual, partial(create_individual, hammetts=hammetts, symmetry_rules=symmetries, prefered_hammetts=None))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover, hammetts=hammetts, symmetry_rules=symmetries, use_continuous_hammett=use_continuous_hammett)
    toolbox.register("mutate", mutate, hammetts=hammetts, symmetry_rules=symmetries, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)
    
    # Execution
    pop = toolbox.population(n=500)
    halloffame = tools.HallOfFame(maxsize=10)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.7, ngen=5,
                        halloffame=halloffame, verbose=True)
    return halloffame, model_s1, model_t1

if __name__ == "__main__":
    # Hammett constants
    hammetts = [0.0, 0.06, 0.66, -0.66, -0.17, -0.37, 0.78, -0.27, 0.45, 0.54, 0.42]
    #hammetts = [0.0, 0.06, 0.66, -0.66, -0.17, -0.37, 0.78] # Original set
    # Whether to use continuous hammett constants
    use_continuous_hammett = False

    # Symmetry rules (Make sure they are correct!!!)
    #symmetries = [(4, 9), (0, 8)]
    all_symmetry_rules = [[(1, 6)],
                          [(0, 2)],
                          [(3, 9)],
                          [(4, 8)],
                          [(5, 7)],
                          [(1, 6), (0, 2)],
                          [(1, 6), (3, 9)],
                          [(1, 6), (4, 8)],
                          [(1, 6), (5, 7)],
                          [(0, 2), (3, 9)],
                          [(0, 2), (4, 8)],
                          [(0, 2), (5, 7)],
                          [(3, 9), (4, 8)],
                          [(3, 9), (5, 7)],
                          [(4, 8), (5, 7)]]
    N_RUNS = 5
    for run in range(1, N_RUNS+1):
        print(f"Starting run {run}")
        with open(f"Symmetric-GA-extended-hammett-continuous_run_{run}.txt", 'w') as f:
            for i, symmetries in enumerate(all_symmetry_rules):
        
                halloffame, model_s1, model_t1 = main(hammetts, symmetry_rules=symmetries, use_continuous_hammett=use_continuous_hammett)
            
                # Results
                print("\n=== Best individuals ===")
                output = f"Symmetries: {' and '.join(str(s) for s in symmetries)}. Observation: 0-indexed"
                print(output)
                f.write((i!=0 and "\n\n" or "")+output+"\n\n")
                for ind in halloffame:
                    lengths = torch.tensor([len(ind)])
                    input_sequence = torch.tensor(ind, dtype=torch.float).view(1, len(ind), 1)
                    S1 = float(model_s1(input_sequence, lengths))
                    T1 = float(model_t1(input_sequence, lengths))
                    output = f"{list(map(float, ind))} -> S1={S1:.3f}, T1={T1:.3f}, S1-T1={S1-T1:.3f}, Fitness={ind.fitness.values[0]:.10f}"
                    print(output)
                    f.write(output+"\n")
