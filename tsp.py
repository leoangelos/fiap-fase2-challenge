import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import json
import os
import numpy as np
import pygame
from benchmark_att48 import *


# Definição de constantes
# pygame
LARGURA, ALTURA = 800, 400
RAIO_NO = 10
FPS = 30
DESLOCAMENTO_X_GRAFICO = 450

# Algoritmo Genético
N_CIDADES = 15
TAMANHO_POPULACAO = 100
N_GERACOES = None
PROBABILIDADE_MUTACAO = 0.5
GERACOES_SEM_MELHORA_PARA_PARAR = 800  # Encerra se não melhorar após N gerações

# Definição de cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
AZUL = (0, 0, 255)


# Inicialização do problema
# Usando geração aleatória de cidades
#localizacoes_cidades = [(random.randint(RAIO_NO + DESLOCAMENTO_X_GRAFICO, LARGURA - RAIO_NO), random.randint(RAIO_NO, ALTURA - RAIO_NO))
#                    for _ in range(N_CIDADES)]


# # Usando problemas padrão: 10, 12 ou 15
# LARGURA, ALTURA = 800, 400
# localizacoes_cidades = default_problems[15]


# Usando benchmark att48
LARGURA, ALTURA = 1500, 800
localizacoes_cidades_att = np.array(att_48_cities_locations)
max_x = max(ponto[0] for ponto in localizacoes_cidades_att)
max_y = max(ponto[1] for ponto in localizacoes_cidades_att)
escala_x = (LARGURA - DESLOCAMENTO_X_GRAFICO - RAIO_NO) / max_x
escala_y = ALTURA / max_y
localizacoes_cidades = [(int(ponto[0] * escala_x + DESLOCAMENTO_X_GRAFICO),
                     int(ponto[1] * escala_y)) for ponto in localizacoes_cidades_att]
solucao_alvo = [localizacoes_cidades[i-1] for i in att_48_cities_order]
fitness_solucao_alvo = calculate_fitness(solucao_alvo)
print(f"Melhor Solução: {fitness_solucao_alvo}")
# ----- Fim benchmark att48


# Inicializar Pygame
pygame.init()
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Solucionador TSP usando Pygame")
relogio = pygame.time.Clock()
contador_geracoes = itertools.count(start=1)  # Iniciar o contador em 1


# --- Heurística do Vizinho Mais Próximo --- (Foi o menos performático no att_48_cities_order)
# def vizinho_mais_proximo(cidades, indice_inicial=0):
#     #Gera uma rota usando a heurística de vizinho mais próximo.
#     nao_visitadas = list(cidades)
#     atual = nao_visitadas.pop(indice_inicial)
#     rota = [atual]
#     while nao_visitadas:
#         mais_proximo = min(nao_visitadas, key=lambda cidade: (cidade[0] - atual[0])**2 + (cidade[1] - atual[1])**2)
#         nao_visitadas.remove(mais_proximo)
#         atual = mais_proximo
#         rota.append(atual)
#     return rota

# solucoes_vmp = []
# for i in range(min(len(localizacoes_cidades), TAMANHO_POPULACAO)):
#     solucoes_vmp.append(vizinho_mais_proximo(localizacoes_cidades, indice_inicial=i))

# restante = TAMANHO_POPULACAO - len(solucoes_vmp)
# solucoes_aleatorias = generate_random_population(localizacoes_cidades, restante) if restante > 0 else []
# populacao = solucoes_vmp + solucoes_aleatorias
# --- Fim Vizinho Mais Próximo ---


# Heurística da Convex Hull
def produto_vetorial(O, A, B):
    #Produto vetorial OA x OB. Positivo se anti-horário.
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def envoltoria_convexa(pontos):
    #Retorna a envoltória convexa dos pontos (algoritmo de Andrew).
    pontos_ordenados = sorted(set(pontos))
    if len(pontos_ordenados) <= 1:
        return pontos_ordenados

    # Parte inferior
    inferior = []
    for p in pontos_ordenados:
        while len(inferior) >= 2 and produto_vetorial(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)

    # Parte superior
    superior = []
    for p in reversed(pontos_ordenados):
        while len(superior) >= 2 and produto_vetorial(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)

    return inferior[:-1] + superior[:-1]


def distancia(a, b):
    #Distância euclidiana entre dois pontos.
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def insercao_envoltoria_convexa(cidades):
    #Constrói uma rota começando pelos pontos externos (Envoltória Convexa)
    #inserindo os pontos internos na posição de menor custo.
    envoltoria = envoltoria_convexa(cidades)
    rota = list(envoltoria)
    restantes = [c for c in cidades if c not in rota]

    for cidade in restantes:
        melhor_aumento = float('inf')
        melhor_pos = 0
        for i in range(len(rota)):
            j = (i + 1) % len(rota)
            aumento = distancia(rota[i], cidade) + distancia(cidade, rota[j]) - distancia(rota[i], rota[j])
            if aumento < melhor_aumento:
                melhor_aumento = aumento
                melhor_pos = j
        rota.insert(melhor_pos, cidade)

    return rota

# Fim - Heurística da Convex Hull


# Início - Criar População Inicial usando heurística do Vizinho Mais Próximo
# solucoes_vmp = []
# for i in range(min(len(localizacoes_cidades), TAMANHO_POPULACAO)):
#    solucoes_vmp.append(vizinho_mais_proximo(localizacoes_cidades, indice_inicial=i))

# restante = TAMANHO_POPULACAO - len(solucoes_vmp)
# solucoes_aleatorias = generate_random_population(localizacoes_cidades, restante) if restante > 0 else []
# populacao = solucoes_vmp + solucoes_aleatorias
# Fim - População usando Vizinho Mais Próximo

# Início - Criar População Inicial usando heurística da Convex Hull
solucao_envoltoria = insercao_envoltoria_convexa(localizacoes_cidades)
fitness_envoltoria = calculate_fitness(solucao_envoltoria)
print(f"Fitness da solução inicial Envoltória Convexa: {round(fitness_envoltoria, 2)}")

populacao = [solucao_envoltoria] + generate_random_population(localizacoes_cidades, TAMANHO_POPULACAO - 1)
# Fim - Criar População Inicial usando Convex Hull

melhores_fitness = []
melhores_solucoes = []
melhor_fitness_global = float('inf')
geracoes_sem_melhora = 0


# Loop principal
executando = True
while executando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            executando = False
        elif evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_q:
                executando = False

    geracao = next(contador_geracoes)

    tela.fill(BRANCO)

    fitness_populacao = [calculate_fitness(
        individuo) for individuo in populacao]

    populacao, fitness_populacao = sort_population(
        populacao, fitness_populacao)

    melhor_fitness = calculate_fitness(populacao[0])
    melhor_solucao = populacao[0]

    melhores_fitness.append(melhor_fitness)
    melhores_solucoes.append(melhor_solucao)

    draw_plot(tela, list(range(len(melhores_fitness))),
              melhores_fitness, y_label="Fitness - Distância (pxls)")

    draw_cities(tela, localizacoes_cidades, VERMELHO, RAIO_NO)
    draw_paths(tela, melhor_solucao, AZUL, width=3)
    draw_paths(tela, populacao[1], rgb_color=(128, 128, 128), width=1)

    print(f"Geração {geracao}: Melhor fitness = {round(melhor_fitness, 2)}")

    # Verificar se houve melhora
    if melhor_fitness < melhor_fitness_global:
        melhor_fitness_global = melhor_fitness
        geracoes_sem_melhora = 0
    else:
        geracoes_sem_melhora += 1

    # Parada automática por estagnação
    if geracoes_sem_melhora >= GERACOES_SEM_MELHORA_PARA_PARAR:
        print(f"\nParada automática: {GERACOES_SEM_MELHORA_PARA_PARAR} gerações sem melhora.")
        executando = False

    nova_populacao = [populacao[0]]  # Manter o melhor indivíduo: ELITISMO

    while len(nova_populacao) < TAMANHO_POPULACAO:

        # seleção
        # seleção simples baseada nas 10 melhores soluções
        # pai1, pai2 = random.choices(populacao[:10], k=2)

        # seleção baseada na probabilidade do fitness
        probabilidade = 1 / np.array(fitness_populacao)
        pai1, pai2 = random.choices(populacao, weights=probabilidade, k=2)

        # filho1 = order_crossover(pai1, pai2)
        filho1 = order_crossover(pai1, pai1)
        filho2 = order_crossover(pai2, pai2)

        filho1 = mutate(filho1, PROBABILIDADE_MUTACAO)
        filho2 = mutate(filho2, PROBABILIDADE_MUTACAO)

        nova_populacao.append(filho1)
        nova_populacao.append(filho2)

    populacao = nova_populacao

    pygame.display.flip()
    relogio.tick(FPS)


# Salvar o melhor indivíduo em arquivo se for melhor que o salvo anteriormente
ARQUIVO_MELHOR_SOLUCAO = "melhor_solucao.json"

melhor_fitness_final = calculate_fitness(melhores_solucoes[-1])
melhor_solucao_final = melhores_solucoes[-1]

salvar = True
if os.path.exists(ARQUIVO_MELHOR_SOLUCAO):
    with open(ARQUIVO_MELHOR_SOLUCAO, 'r') as f:
        dados_salvos = json.load(f)
        fitness_salvo = dados_salvos.get('fitness', float('inf'))
        if melhor_fitness_final >= fitness_salvo:
            salvar = False
            print(f"Solução atual ({round(melhor_fitness_final, 2)}) não é melhor que a salva ({round(fitness_salvo, 2)}). Não salvando.")

if salvar:
    dados = {
        'fitness': melhor_fitness_final,
        'rota': melhor_solucao_final,
        'numero_geracoes': geracao
    }
    with open(ARQUIVO_MELHOR_SOLUCAO, 'w') as f:
        json.dump(dados, f, indent=2)
    print(f"Nova melhor solução salva! Fitness: {round(melhor_fitness_final, 2)} em {geracao} gerações.")

# encerrar programa
pygame.quit()
sys.exit()
