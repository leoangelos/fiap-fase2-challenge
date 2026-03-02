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

# Algoritmo Genético - mTSP
N_CIDADES = 15
TAMANHO_POPULACAO = 100
N_GERACOES = None
PROBABILIDADE_MUTACAO = 0.5
GERACOES_SEM_MELHORA_PARA_PARAR = 800  # Encerra se não melhorar após N gerações
N_VEICULOS = 3  # Número de veículos
HEURISTICA = 1  # 1 = Vizinho Mais Próximo | 2 = Convex Hull

# Definição de cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
AZUL = (0, 0, 255)
VERDE = (0, 180, 0)
AMARELO = (200, 200, 0)

# Cores para cada veículo (uma por rota)
CORES_VEICULOS = [
    (0, 0, 255),       # Azul
    (0, 180, 0),       # Verde
    (200, 130, 0),     # Laranja
    (180, 0, 180),     # Roxo
    (0, 180, 180),     # Ciano
]


# Usando benchmark att48
LARGURA, ALTURA = 1500, 800
localizacoes_cidades_att = np.array(att_48_cities_locations)
max_x = max(ponto[0] for ponto in localizacoes_cidades_att)
max_y = max(ponto[1] for ponto in localizacoes_cidades_att)
escala_x = (LARGURA - DESLOCAMENTO_X_GRAFICO - RAIO_NO) / max_x
escala_y = ALTURA / max_y
localizacoes_cidades = [(int(ponto[0] * escala_x + DESLOCAMENTO_X_GRAFICO),
                     int(ponto[1] * escala_y)) for ponto in localizacoes_cidades_att]

# Depósito: primeira cidade do benchmark
DEPOSITO = localizacoes_cidades[0]
# Cidades a visitar (todas exceto o depósito)
cidades_sem_deposito = [c for c in localizacoes_cidades if c != DEPOSITO]

print(f"Depósito: {DEPOSITO}")
print(f"Número de cidades a visitar: {len(cidades_sem_deposito)}")
print(f"Número de veículos: {N_VEICULOS}")
# ----- Fim benchmark att48


# === Funções do mTSP ===

def dividir_rota(cromossomo, n_veiculos):
    """Divide um cromossomo (permutação de cidades) em N sub-rotas equilibradas."""
    n_cidades = len(cromossomo)
    tamanho_base = n_cidades // n_veiculos
    resto = n_cidades % n_veiculos

    rotas = []
    inicio = 0
    for i in range(n_veiculos):
        # Distribui o resto entre os primeiros veículos
        tamanho = tamanho_base + (1 if i < resto else 0)
        rotas.append(cromossomo[inicio:inicio + tamanho])
        inicio += tamanho

    return rotas


def calcular_fitness_mtsp(cromossomo, deposito, n_veiculos):
    """Calcula o fitness total do mTSP: soma das distâncias de todos os veículos.
    Cada veículo faz: depósito → cidades atribuídas → depósito."""
    rotas = dividir_rota(cromossomo, n_veiculos)
    distancia_total = 0

    for rota in rotas:
        if len(rota) == 0:
            continue
        # Depósito → primeira cidade
        distancia_total += distancia_euclidiana(deposito, rota[0])
        # Percurso entre cidades
        for i in range(len(rota) - 1):
            distancia_total += distancia_euclidiana(rota[i], rota[i + 1])
        # Última cidade → depósito
        distancia_total += distancia_euclidiana(rota[-1], deposito)

    return distancia_total


def distancia_euclidiana(a, b):
    """Distância euclidiana entre dois pontos."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def ordenar_populacao_mtsp(populacao, deposito, n_veiculos):
    """Ordena a população pelo fitness mTSP (menor é melhor)."""
    fitness_lista = [calcular_fitness_mtsp(ind, deposito, n_veiculos) for ind in populacao]
    pares = list(zip(populacao, fitness_lista))
    pares.sort(key=lambda x: x[1])
    populacao_ordenada = [p[0] for p in pares]
    fitness_ordenado = [p[1] for p in pares]
    return populacao_ordenada, fitness_ordenado


# --- Heurística da Convex Hull adaptada para mTSP ---
def produto_vetorial(O, A, B):
    #Produto vetorial OA x OB. Positivo se anti-horário.
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

def envoltoria_convexa(pontos):
    #Retorna a Convex Hull dos pontos (algoritmo de Andrew).
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

def insercao_envoltoria_convexa(cidades):
    #Constrói uma rota começando pelos pontos externos (Convex Hull)
    envoltoria = envoltoria_convexa(cidades)
    rota = list(envoltoria)
    restantes = [c for c in cidades if c not in rota]
    for cidade in restantes:
        melhor_aumento = float('inf')
        melhor_pos = 0
        for i in range(len(rota)):
            j = (i + 1) % len(rota)
            aumento = distancia_euclidiana(rota[i], cidade) + distancia_euclidiana(cidade, rota[j]) - distancia_euclidiana(rota[i], rota[j])
            if aumento < melhor_aumento:
                melhor_aumento = aumento
                melhor_pos = j
        rota.insert(melhor_pos, cidade)
    return rota
# --- Fim Convex Hull ---


# --- Heurística do Vizinho Mais Próximo adaptada para mTSP ---
def vizinho_mais_proximo(cidades, indice_inicial=0):
    #Gera uma rota usando a heurística de vizinho mais próximo.
    nao_visitadas = list(cidades)
    atual = nao_visitadas.pop(indice_inicial)
    rota = [atual]
    while nao_visitadas:
        mais_proximo = min(nao_visitadas, key=lambda cidade: (cidade[0] - atual[0])**2 + (cidade[1] - atual[1])**2)
        nao_visitadas.remove(mais_proximo)
        atual = mais_proximo
        rota.append(atual)
    return rota
# --- Fim Vizinho Mais Próximo ---


# === Gerar População Aleatória (sem depósito) ===
def gerar_populacao_aleatoria_mtsp(cidades, tamanho_populacao):
    """Gera população aleatória de permutações das cidades (sem o depósito)."""
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = list(cidades)
        random.shuffle(individuo)
        populacao.append(individuo)
    return populacao


# === Criar População Inicial baseada na HEURISTICA escolhida ===
if HEURISTICA == 1:
    # Vizinho Mais Próximo
    print("Heurística selecionada: Vizinho Mais Próximo")
    solucoes_vmp = []
    for i in range(min(len(cidades_sem_deposito), TAMANHO_POPULACAO)):
        solucoes_vmp.append(vizinho_mais_proximo(cidades_sem_deposito, indice_inicial=i))
    restante = TAMANHO_POPULACAO - len(solucoes_vmp)
    solucoes_aleatorias = gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, restante) if restante > 0 else []
    populacao = solucoes_vmp + solucoes_aleatorias

elif HEURISTICA == 2:
    # Convex Hull
    print("Heurística selecionada: Convex Hull")
    solucao_envoltoria = insercao_envoltoria_convexa(cidades_sem_deposito)
    populacao = [solucao_envoltoria] + gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, TAMANHO_POPULACAO - 1)

else:
    print("Heurística inválida! Usando população aleatória.")
    populacao = gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, TAMANHO_POPULACAO)

fitness_inicial = calcular_fitness_mtsp(populacao[0], DEPOSITO, N_VEICULOS)
print(f"Fitness da melhor solução inicial (mTSP): {round(fitness_inicial, 2)}")


# Inicializar Pygame
pygame.init()
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption(f"mTSP - {N_VEICULOS} Veículos usando Pygame")
relogio = pygame.time.Clock()
contador_geracoes = itertools.count(start=1)  # Iniciar o contador em 1

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

    # Calcular fitness e ordenar população
    populacao, fitness_populacao = ordenar_populacao_mtsp(populacao, DEPOSITO, N_VEICULOS)

    melhor_fitness = fitness_populacao[0]
    melhor_solucao = populacao[0]

    melhores_fitness.append(melhor_fitness)
    melhores_solucoes.append(melhor_solucao)

    # Desenhar gráfico de convergência
    draw_plot(tela, list(range(len(melhores_fitness))),
              melhores_fitness, y_label="Fitness - Distância (pxls)")

    # Desenhar cidades e depósito
    draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
    # Depósito com cor especial e tamanho maior
    pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
    pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)

    # Desenhar rotas da melhor solução (uma cor por veículo)
    rotas_melhor = dividir_rota(melhor_solucao, N_VEICULOS)
    for idx, rota in enumerate(rotas_melhor):
        if len(rota) < 2:
            if len(rota) == 1:
                # Desenha linha depósito → cidade → depósito
                pygame.draw.line(tela, CORES_VEICULOS[idx % len(CORES_VEICULOS)], DEPOSITO, rota[0], 3)
                pygame.draw.line(tela, CORES_VEICULOS[idx % len(CORES_VEICULOS)], rota[0], DEPOSITO, 3)
            continue
        cor = CORES_VEICULOS[idx % len(CORES_VEICULOS)]
        # Rota completa: depósito → cidades → depósito
        rota_completa = [DEPOSITO] + list(rota) + [DEPOSITO]
        draw_paths(tela, rota_completa, cor, width=3)

    # Desenhar rota do segundo melhor (cinza, para referência)
    if len(populacao) > 1:
        rotas_segundo = dividir_rota(populacao[1], N_VEICULOS)
        for rota in rotas_segundo:
            if len(rota) >= 2:
                rota_completa = [DEPOSITO] + list(rota) + [DEPOSITO]
                draw_paths(tela, rota_completa, (180, 180, 180), width=1)

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

        # seleção baseada na probabilidade do fitness
        probabilidade = 1 / np.array(fitness_populacao)
        pai1, pai2 = random.choices(populacao, weights=probabilidade, k=2)

        # crossover e mutação
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
ARQUIVO_MELHOR_SOLUCAO = "melhor_solucao_mtsp.json"

melhor_fitness_final = calcular_fitness_mtsp(melhores_solucoes[-1], DEPOSITO, N_VEICULOS)
melhor_solucao_final = melhores_solucoes[-1]
rotas_finais = dividir_rota(melhor_solucao_final, N_VEICULOS)

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
        'n_veiculos': N_VEICULOS,
        'deposito': DEPOSITO,
        'rotas': [list(rota) for rota in rotas_finais],
        'numero_geracoes': geracao
    }
    with open(ARQUIVO_MELHOR_SOLUCAO, 'w') as f:
        json.dump(dados, f, indent=2)
    print(f"Nova melhor solução mTSP salva! Fitness: {round(melhor_fitness_final, 2)} em {geracao} gerações.")

# Imprimir resumo das rotas
print(f"\n=== Resumo mTSP ({N_VEICULOS} veículos) ===")
for idx, rota in enumerate(rotas_finais):
    dist = distancia_euclidiana(DEPOSITO, rota[0]) if rota else 0
    for i in range(len(rota) - 1):
        dist += distancia_euclidiana(rota[i], rota[i + 1])
    if rota:
        dist += distancia_euclidiana(rota[-1], DEPOSITO)
    print(f"Veículo {idx + 1}: {len(rota)} cidades, distância = {round(dist, 2)}")
print(f"Distância total: {round(melhor_fitness_final, 2)}")

# encerrar programa
pygame.quit()
sys.exit()
