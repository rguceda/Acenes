#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:20:21 2025

@author: Boris Pérez-Cañedo
"""
import operator
import random
from itertools import combinations
import numpy
import math
from deap import base
from deap import creator
from deap import tools
from GA import main

# Hammett constants
hammetts = [0.0, 0.06, 0.66, -0.66, -0.17, -0.37, 0.78, -0.27, 0.45, 0.54, 0.42]

# Recurrent Neural Network
import torch 
from RNN import GRURegressor

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
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def repair(part, max_nonzeros=4):
    ind = list(part)
    # Get indices and values of non-zero elements
    nonzero_info = [(i, abs(v)) for i, v in enumerate(ind) if v != 0]
    
    # If there are more than max_nonzeros, convert the ones closest to zero to 0
    if len(nonzero_info) > max_nonzeros:
        # Sort by absolute value (closest to zero first)
        nonzero_info.sort(key=lambda x: x[1])
        
        # Calculate how many we need to remove
        extra = len(nonzero_info) - max_nonzeros
        
        # Set the 'extra' values closest to zero to 0
        for i, _ in nonzero_info[:extra]:
            ind[i] = 0
    
    # Ensure it's not all zeros (at least 1 position with non-zero value)
    if all(v == 0 for v in ind):
        idx = numpy.random.randint(len(ind))
        # Assuming hammetts is defined elsewhere - you might want to pass it as parameter
        valores = [v for v in hammetts if v != 0]
        ind[idx] = numpy.random.choice(valores)
    
    return ind

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    part[:] = repair(part)

def evaluate(ind, discrete_sol, positions):
    lengths = torch.tensor([len(discrete_sol)])
    solution = list(discrete_sol)
    for i, pos in enumerate(positions):
        solution[pos] = ind[i]
    input_sequence = torch.tensor(solution, dtype=torch.float).view(1, len(solution), 1)
    S1 = float(model_s1(input_sequence, lengths))
    T1 = float(model_t1(input_sequence, lengths))
    # Penalty if S1 < 2*T1
    penalty = max(0, 2*T1 - S1)
    return (-penalty, )

def main_pso(discrete_sol, positions):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=len(positions), pmin=min(hammetts), pmax=max(hammetts), smin=-0.1, smax=0.1)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", evaluate, discrete_sol=discrete_sol, positions=positions)
    pop = toolbox.population(n=50)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 100
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    best_sol = list(discrete_sol)
    for i, pos in enumerate(positions):
        best_sol[pos] = best[i]
    return best_sol, best.fitness.values[0]

N_RUNS = 10
number_of_var_pos = 1 # The number of non-zero Hammett constants in the GA solution that PSO can change
for run in range(1, N_RUNS+1):
    print(f"Starting run {run}")  
    with open(f"PSO_multi_dim_{number_of_var_pos}_run_{run}.txt", 'w') as f:
        # Number of trials
        N = 1
        for _ in range(N):
            halloffame, _, _ = main(hammetts)
            for ind in halloffame:
                lengths = torch.tensor([len(ind)])
                input_sequence = torch.tensor(ind, dtype=torch.float).view(1, len(ind), 1)
                S1 = float(model_s1(input_sequence, lengths))
                T1 = float(model_t1(input_sequence, lengths))
                f.writelines(["Genetic solution\n\n", ", ".join([f"{val:.3f}" for val in ind]) + f", S1={S1:.3f}, T1={T1:.3f}, S1-T1={S1-T1:.3f}, Fitness={max(0, 2*T1 - S1):.3f}\n\n"])
                nonzero_idx = [i for i, v in enumerate(ind) if v != 0]
                positions = [list(c) for c in combinations(nonzero_idx, number_of_var_pos)]
                f.write("Corresponding PSO solutions\n\n")
                for pos in positions:
                    best, fitness = main_pso(ind, pos)
                    lengths = torch.tensor([len(best)])
                    input_sequence = torch.tensor(best, dtype=torch.float).view(1, len(best), 1)
                    S1 = float(model_s1(input_sequence, lengths))
                    T1 = float(model_t1(input_sequence, lengths))
                    f.write(", ".join([f"{val:.3f}" for val in best]) + f", S1={S1:.3f}, T1={T1:.3f}, S1-T1={S1-T1:.3f}, Fitness={fitness:.5f}\n")
    print(f"Ended run {run}")
