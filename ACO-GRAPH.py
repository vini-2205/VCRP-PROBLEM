import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import time

# Parâmetros do ACO
alpha = 0.9
beta = 5.0
sigma = 2
rho = 0.85
theta = 70
reiniciar_a_cada = 200
iterations = 10000  # Número de iterações
num_formigas = 40  # Número de formigas
num_caminhoes = 5  # Número de caminhões disponíveis
iteracoes_list = [500,1000,2500,5000]

# Função para ler o arquivo .vrp e extrair as informações
def ler_arquivo_vrp(arquivo):
    with open(arquivo, 'r') as file:
        linhas = file.readlines()

    secao_nos = False
    secao_demanda = False
    coordenadas = {}
    demandas = {}
    capacidade = None
    valor_otimo = None

    for linha in linhas:
        if "CAPACITY" in linha:
            capacidade = int(linha.split(":")[1].strip())
        elif "OPTIMAL" in linha:
            valor_otimo = int(linha.split(":")[1].strip())
        elif "NODE_COORD_SECTION" in linha:
            secao_nos = True
            secao_demanda = False
        elif "DEMAND_SECTION" in linha:
            secao_demanda = True
            secao_nos = False
        elif secao_nos:
            if "EOF" in linha or "DEMAND_SECTION" in linha:
                secao_nos = False
            else:
                no, x, y = map(int, linha.strip().split())
                coordenadas[no] = (x, y)
        elif secao_demanda:
            if "DEPOT_SECTION" in linha:
                break
            else:
                no, demanda = map(int, linha.strip().split())
                demandas[no] = demanda

    return coordenadas, demandas, capacidade

# Função para gerar o grafo (arestas e feromônios)
def gerar_grafo(coordenadas):
    vertices = list(coordenadas.keys())
    vertices.remove(1)  # Remover o depósito (nó 1)
    arestas = {
        (min(a, b), max(a, b)): np.sqrt((coordenadas[a][0] - coordenadas[b][0]) ** 2 + 
                                        (coordenadas[a][1] - coordenadas[b][1]) ** 2)
        for a in coordenadas.keys() for b in coordenadas.keys() if a != b
    }

    # Inicializar feromônios com base no valor inverso das distâncias médias
    media_distancia = np.mean(list(arestas.values()))
    feromonios = {k: 1 / (len(vertices) * media_distancia) for k in arestas.keys()}
    
    return vertices, arestas, feromonios


# Função para avaliar uma solução
def avaliar_solucao(solucao, arestas):
    custo = 0
    for rota in solucao:
        a = 1  # Depósito
        for b in rota:
            custo += arestas[(min(a, b), max(a, b))]
            a = b
        custo += arestas[(min(a, 1), max(a, 1))]  # Volta ao depósito
    return custo

# Função para atualizar os feromônios
def atualizar_feromonio(feromonios, solucoes, melhor_solucao):
    custo_medio = np.mean([s[1] for s in solucoes])
    for k in feromonios:
        feromonios[k] *= (1 - rho)  # Evaporação

    melhor_solucao_local = min(solucoes, key=lambda x: x[1])
    
    if melhor_solucao is None:
        melhor_solucao = melhor_solucao_local
    else:
        melhor_solucao = min(melhor_solucao, melhor_solucao_local, key=lambda x: x[1])

    for rota in melhor_solucao[0]:
        for i in range(len(rota) - 1):
            feromonios[(min(rota[i], rota[i + 1]), max(rota[i + 1], rota[i]))] += sigma / melhor_solucao[1]

    return melhor_solucao

def solucao_uma_formiga(vertices, arestas, capacidade_limite, demanda, feromonios, num_caminhoes, formiga, iteracao):
    solucao = []
    cidades_restantes = set(vertices)  # Manter controle das cidades restantes
    caminhões_usados = 0

    while cidades_restantes and caminhões_usados < num_caminhoes:
        rota = []
        capacidade_restante = capacidade_limite
        cidade_atual = 1  # Começa no depósito (nó 1)

        while cidades_restantes:
            probabilidades = [
                (feromonios.get((min(cidade_atual, x), max(cidade_atual, x)), 0) ** alpha) * 
                ((1 / arestas.get((min(cidade_atual, x), max(cidade_atual, x)), float('inf'))) ** beta)
                for x in cidades_restantes
            ]

            soma_probabilidades = np.sum(probabilidades)
            if soma_probabilidades == 0:
                probabilidades = np.ones(len(cidades_restantes)) / len(cidades_restantes)
            else:
                probabilidades /= soma_probabilidades

            proxima_cidade = np.random.choice(list(cidades_restantes), p=probabilidades)

            if capacidade_restante - demanda[proxima_cidade] >= 0:
                capacidade_restante -= demanda[proxima_cidade]
                rota.append(proxima_cidade)
                cidades_restantes.remove(proxima_cidade)
                cidade_atual = proxima_cidade
            else:
                break

        if rota:  # Adicionar a rota se houver cidades visitadas
            solucao.append(rota)
            caminhões_usados += 1

    if cidades_restantes:
        # Se houver cidades restantes, a solução não é válida
        return None
    
    return solucao

# Função para gerar gráfico das rotas
def salvar_grafico(filename):
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    plt.savefig(os.path.join('resultados', filename))
    plt.clf()  # Limpar o gráfico atual para a próxima plotagem

# Função para gerar gráfico das rotas
def gerar_grafico(coordenadas, melhor_solucao, iteracoes):
    for rota in melhor_solucao[0]:
        x = [coordenadas[1][0]] + [coordenadas[cidade][0] for cidade in rota] + [coordenadas[1][0]]
        y = [coordenadas[1][1]] + [coordenadas[cidade][1] for cidade in rota] + [coordenadas[1][1]]
        plt.plot(x, y, marker='o')
    
    plt.title(f'Melhor Solução - {iteracoes} Iterações')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    salvar_grafico(f'melhor_solucao_{iteracoes}_iteracoes.png')

# Função para gerar gráfico do custo por iteração
def gerar_grafico_custo_por_iteracao(custos_por_iteracao, iteracoes):
    plt.plot(custos_por_iteracao, marker='o')
    plt.title(f'Custo por Iteração - {iteracoes} Iterações')
    plt.xlabel('Iteração')
    plt.ylabel('Custo')
    plt.grid(True)
    salvar_grafico(f'custo_por_iteracao_{iteracoes}_iteracoes.png')

def rodar_aco_para_variadas_iteracoes():
    coordenadas, demanda, capacidade_limite = ler_arquivo_vrp('test-n18.vrp')
    
    for iteracoes in iteracoes_list:
        melhor_solucao = None
        iteracoes_sem_melhora = 0
        vertices, arestas, feromonios = gerar_grafo(coordenadas)

        custos_por_iteracao = []

        start_time = time.time()  # Início do tempo

        for i in range(iteracoes):
            solucoes = []
            for j in range(num_formigas):
                solucao = solucao_uma_formiga(vertices.copy(), arestas, capacidade_limite, demanda, feromonios, num_caminhoes, j, i)
                
                if solucao is not None:
                    custo = avaliar_solucao(solucao, arestas)
                    solucoes.append((solucao, custo))

            if solucoes:  # Verifica se há soluções válidas
                melhor_solucao_anterior = melhor_solucao
                melhor_solucao = atualizar_feromonio(feromonios, solucoes, melhor_solucao)

                # Armazena o melhor custo para essa iteração
                custos_por_iteracao.append(melhor_solucao[1])

                # Verificar se houve melhora
                if melhor_solucao_anterior is None or melhor_solucao[1] < melhor_solucao_anterior[1]:
                    iteracoes_sem_melhora = 0
                else:
                    iteracoes_sem_melhora += 1

                # Reiniciar feromônios se estagnar
                if iteracoes_sem_melhora >= reiniciar_a_cada:
                    vertices, arestas, feromonios = gerar_grafo(coordenadas)
                    iteracoes_sem_melhora = 0
                    print(f"Reiniciando feromônios na iteração {i + 1} ({iteracoes} Iterações)")

                print(f"Iteração {i + 1}: Melhor custo encontrado = {int(melhor_solucao[1])} ({iteracoes} Iterações)")
        
        end_time = time.time()  # Fim do tempo
        execution_time = end_time - start_time
        print(f"Tempo de execução para {iteracoes} iterações: {execution_time:.2f} segundos")

        if melhor_solucao:  # Verifica se há uma solução antes de salvar os resultados
            # Salva os gráficos e resultados
            gerar_grafico(coordenadas, melhor_solucao, iteracoes)
            gerar_grafico_custo_por_iteracao(custos_por_iteracao, iteracoes)

            # Salvar resultados em arquivo de texto
            with open(os.path.join('resultados', f'resultados_{iteracoes}_iteracoes.txt'), 'w') as f:
                f.write(f"Melhor solução final: {melhor_solucao}\n")
                f.write(f"Custo final: {melhor_solucao[1]}\n")
                f.write(f"Tempo de execução: {execution_time:.2f} segundos\n")

if __name__ == "__main__":
    rodar_aco_para_variadas_iteracoes()