import os
import time
from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt

def read_vrp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    demands = []
    capacity = 0
    dimension = 0
    section = None

    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(line.split()[-1])
        elif line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            section = "NODE_COORD_SECTION"
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMAND_SECTION"
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT_SECTION"
        elif line.startswith("EOF") or line.startswith(" -1"):
            break
        elif section == "NODE_COORD_SECTION":
            _, x, y = map(float, line.split())
            coordinates.append((x, y))
        elif section == "DEMAND_SECTION":
            _, demand = map(int, line.split())
            demands.append(demand)

    return coordinates, demands, capacity, dimension

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def save_results_and_plot(coordinates, solution, cost, exec_time, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Salvando o gráfico das rotas
    plt.figure(figsize=(10, 8))
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if solution[i, j] > 0.5:
                plt.plot([coordinates[i][0], coordinates[j][0]], [coordinates[i][1], coordinates[j][1]], 'b-')
    for coord in coordinates:
        plt.plot(coord[0], coord[1], 'ro')
    plt.title(f'Rotas encontradas (Custo: {cost})')
    plt.savefig(os.path.join(folder_path, 'rotas.png'))
    plt.close()

    # Salvando os resultados em um arquivo txt
    with open(os.path.join(folder_path, 'resultados.txt'), 'w') as f:
        f.write(f'Solução ótima encontrada com custo total: {cost}\n')
        f.write(f'Tempo de execução: {exec_time} segundos\n')
        f.write('Rotas:\n')
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if solution[i, j] > 0.5:
                    f.write(f'Rota de {i} para {j}\n')

# Leitura do arquivo VRP
file_path = 'A-n32-k5.vrp'
coordinates, demands, capacity, dimension = read_vrp_file(file_path)

# Definindo a cidade 1 como o depósito
depot = coordinates[0]
coordinates = coordinates[1:]
coordinates.insert(0, depot)

depot_demand = demands[0]
demands = demands[1:]
demands.insert(0, depot_demand)

# Atualiza n para o número total de nós (incluindo depósito)
n = len(coordinates)

# Matriz de distâncias
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        distance_matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
        distance_matrix[j][i] = distance_matrix[i][j]

# Criação do modelo
model = Model('CVRP')

# Variáveis de decisão: x[i, j] = 1 se o arco (i, j) é utilizado, 0 caso contrário
x = model.addVars(n, n, vtype=GRB.BINARY, name='x')

# Variáveis auxiliares para evitar sub-rotas
u = model.addVars(n, vtype=GRB.INTEGER, name='u') 

# Função objetivo: minimizar a soma das distâncias percorridas
model.setObjective(quicksum(x[i, j] * distance_matrix[i][j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Restrições:
model.addConstrs((quicksum(x[i, j] for j in range(n) if i != j) == 1 for i in range(1, n)), name="visit_once")
model.addConstrs((quicksum(x[i, j] for i in range(n) if i != j) == 1 for j in range(1, n)), name="leave_once")
model.addConstrs((u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j] for i in range(1, n) for j in range(1, n) if i != j), name="capacity")
model.addConstr(quicksum(x[0, j] for j in range(1, n)) <= n - 1, name="start_depot")
model.addConstr(quicksum(x[i, 0] for i in range(1, n)) <= n - 1, name="end_depot")
model.addConstrs((u[i] >= demands[i] for i in range(1, n)), name="min_demand")
model.addConstrs((u[i] <= capacity for i in range(1, n)), name="max_capacity")

# Limite do número de caminhões (k)
k = 5  # Número de caminhões
model.addConstr(quicksum(x[i, j] for i in range(n) for j in range(n) if i != j) <= k * (n - 1), name="limit_trucks")

# Otimizar o modelo
start_time = time.time()
model.optimize()
end_time = time.time()

exec_time = end_time - start_time

# Verificar se há solução ótima
if model.status == GRB.OPTIMAL:
    print('Solução ótima encontrada com custo total: {}'.format(model.objVal))
    solution = model.getAttr('x', x)
    
    # Salvando os resultados e gráficos
    save_results_and_plot(coordinates, solution, model.objVal, exec_time, 'resultados')
else:
    print('Nenhuma solução ótima encontrada.')
