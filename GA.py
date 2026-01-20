import numpy as np
from deap import base, creator, tools, algorithms

# ======================
# CARGA LOS MODELOS
# ======================
import torch
from RNN import GRURegressor

# Hammett constants
hammetts = [0.0, 0.06, 0.66, -0.66, -0.17, -0.37, 0.78, -0.27, 0.45, 0.54, 0.42]

def main(hammetts):
    
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
    
    
    # ======================
    # CONFIGURAR DEAP
    # ======================
    # Crear tipos
    for name in ["FitnessMin","Individual"]:
        if name in creator.__dict__:
            delattr(creator, name)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Valores posibles por posición
    
    solution_length = 6 # 6 para Benceno, 8 para Naftaleno, 10 para Antraceno, 12 para Tetraceno, 14 para Pentaceno
    
    # ======================
    # FUNCIONES DE INDIVIDUO
    # ======================
    def crear_individuo(max_nonzeros=4):
        n_positions = solution_length
        ind = [0]*n_positions  # empezamos con todo cero
        
        # Elegir cuántas posiciones serán distintas de 0 (1 a max_nonzeros)
        n_nonzeros = np.random.randint(1, max_nonzeros + 1)
        
        # Elegir posiciones que tendrán valores distintos de 0
        nonzero_positions = np.random.choice(range(n_positions), size=n_nonzeros, replace=False)
        
        for i in nonzero_positions:
            valores = [v for v in hammetts if v != 0]
            ind[i] = np.random.choice(valores)
        
        return creator.Individual(ind)
    
    def reparar_individuo(ind, max_nonzeros=4):
        nonzero_idx = [i for i, v in enumerate(ind) if v != 0]
        
        # Si hay más de max_nonzeros, convertimos algunos a cero
        if len(nonzero_idx) > max_nonzeros:
            extra = len(nonzero_idx) - max_nonzeros
            for i in np.random.choice(nonzero_idx, extra, replace=False):
                ind[i] = 0
        
        # Aseguramos que no sea todo cero (al menos 1 posición con valor distinto de 0)
        if all(v == 0 for v in ind):
            idx = np.random.randint(len(ind))
            valores = [v for v in hammetts if v != 0]
            ind[idx] = np.random.choice(valores)
        
        return creator.Individual(ind)
    
    
    def cruzar(ind1, ind2):
        punto = np.random.randint(1, len(ind1))
        new1 = ind1[:punto] + ind2[punto:]
        new2 = ind2[:punto] + ind1[punto:]
        return reparar_individuo(new1), reparar_individuo(new2)
    
    def mutar(ind, indpb=0.3):
        new = []
        for val in ind:
            if np.random.rand() < indpb:
                values = [v for v in hammetts if v != val] # Si ya decidimos mutar, entonces esto evita que sustituya la misma hammett
                new.append(np.random.choice(values))
            else:
                new.append(val)
        return reparar_individuo(new),
    
    def evaluar(ind):
        lengths = torch.tensor([len(ind)])
        input_sequence = torch.tensor(ind, dtype=torch.float).view(1, len(ind), 1)
        S1 = float(model_s1(input_sequence, lengths))
        T1 = float(model_t1(input_sequence, lengths))
        
        # Penalty if S1 < 2*T1
        penalty = max(0, 2*T1 - S1)
        return (penalty,)
    
    # ======================
    # REGISTRO EN TOOLBOX
    # ======================
    toolbox.register("individual", tools.initIterate, creator.Individual, crear_individuo)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cruzar)
    toolbox.register("mutate", mutar, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluar)
    
    # ======================
    # EJECUCIÓN
    # ======================
    pop = toolbox.population(n=500)
    halloffame = tools.HallOfFame(maxsize=10)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.7, ngen=5,
                        halloffame=halloffame, verbose=True)
    return halloffame, model_s1, model_t1

if __name__ == "__main__":
    halloffame, model_s1, model_t1 = main(hammetts)

    # ======================
    # RESULTADOS
    # ======================
    print("\n=== Mejores candidatos ===")
    for ind in halloffame:
        lengths = torch.tensor([len(ind)])
        input_sequence = torch.tensor(ind, dtype=torch.float).view(1, len(ind), 1)
        S1 = float(model_s1(input_sequence, lengths))
        T1 = float(model_t1(input_sequence, lengths))
        print(f"{list(map(float, ind))} -> S1={S1:.3f}, T1={T1:.3f}, S1-T1={S1-T1:.3f}, Fitness={ind.fitness.values[0]:.10f}")
