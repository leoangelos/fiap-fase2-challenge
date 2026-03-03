# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          ALGORITMO GENÉTICO PARA O PROBLEMA DO CAIXEIRO VIAJANTE           ║
# ║              MÚLTIPLO (mTSP - multiple Travelling Salesman Problem)        ║
# ║                                                                            ║
# ║  Codificação Combinatória: Cada cromossomo é uma permutação das cidades    ║
# ║  (sem repetição), dividida em N sub-rotas para N veículos.                 ║
# ║  Codificação Híbrida: combina permutação (ordem das cidades) com partição  ║
# ║  (divisão equilibrada entre veículos).                                     ║
# ║                                                                            ║
# ║  Pipeline do AG:                                                           ║
# ║    1. Gerar População Inicial (heurística ou aleatória)                    ║
# ║    2. Avaliar Aptidão (Fitness)                                            ║
# ║    3. Loop evolutivo:                                                      ║
# ║       a. Validar condição de término                                       ║
# ║       b. Seleção dos pais                                                  ║
# ║       c. Cruzamento (crossover)                                            ║
# ║       d. Mutação                                                           ║
# ║       e. Substituição da população                                         ║
# ║    4. Pós-processamento (rebalanceamento, 2-opt, busca local)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover
from draw_functions import draw_paths, draw_plot, draw_cities, draw_info_overlay
import sys
import json
from datetime import datetime
import os
import numpy as np
from benchmark_att48 import *


# ============================================================================
# PARÂMETROS E CONSTANTES
# ============================================================================

# Pygame (visualização)
LARGURA, ALTURA = 800, 400
RAIO_NO = 10
FPS = 30
DESLOCAMENTO_X_GRAFICO = 450

# Ativar relatório GPT
RELATORIO_GPT = False

# Parâmetros do Algoritmo Genético
N_CIDADES = 48                          # Número de cidades Max 48
TAMANHO_POPULACAO = 100                 # Tamanho da população de cromossomos

PROBABILIDADE_MUTACAO = 0.8             # Probabilidade de mutação (intensidade: alta = explora mais)
GERACOES_SEM_MELHORA_PARA_PARAR = 800   # Critério de convergência: para após N gerações sem melhora
HEURISTICA = 3                          # 1 = Vizinho Mais Próximo | 2 = Convex Hull | 3 = Aleatório
PESO_BALANCEAMENTO = 0.6                # Penalidade por desbalanceamento entre rotas

# Modo de seleção do depósito
# 'primeiro'  → usa a primeira cidade do dataset (comportamento original)
# 'central'   → usa a cidade mais próxima do centróide de todas as cidades
#               (minimiza a distância média de saída/retorno para toda a frota)
DEPOSITO_MODO = 'central'

# Objetivo de otimização
# 'distancia' → minimiza km total percorrido (economia de combustível)
#               Carros (600km autonomia) assumem cidades distantes,
#               motos (300km) ficam com cidades próximas.
# 'tempo'     → equilibra tempo de chegada entre todos os veículos
# 'hibrido'   → combina os dois objetivos: economia de combustível + equilíbrio de tempo
#               Ajuste ALFA_HIBRIDO: 0.0 = só distância, 1.0 = só tempo, 0.5 = balanceado
OBJETIVO = 'distancia'
ALFA_HIBRIDO = 0.5      # Peso do componente de TEMPO no híbrido (0.0 a 1.0)
VELOCIDADE_REF = 100    # km/h — converte horas em km-equivalente para normalizar escalas

# Frota Heterogênea — Tipos de Veículos
N_CARROS = 3                            # Quantidade de carros
N_MOTOS = 0                             # Quantidade de motos
N_VEICULOS = N_CARROS + N_MOTOS         # Total de veículos

VELOCIDADE_CARRO = 100                   # km/h
VELOCIDADE_MOTO = 120                   # km/h
AUTONOMIA_CARRO = 1200                   # km (precisa reabastecer na base)
AUTONOMIA_MOTO = 400                    # km (precisa reabastecer na base)

# Configuração de Operação
REABASTECIMENTO_ATIVO = False            # Se False, veículos ganham autonomia infinita

# Lista de veículos: primeiros N_CARROS são carros, restante são motos
VEICULOS = (
    [{'tipo': 'Carro 🚗', 'velocidade': VELOCIDADE_CARRO, 'autonomia': AUTONOMIA_CARRO}] * N_CARROS +
    [{'tipo': 'Moto 🏍️', 'velocidade': VELOCIDADE_MOTO, 'autonomia': AUTONOMIA_MOTO}] * N_MOTOS
)

# Definição de cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)

AMARELO = (200, 200, 0)

# Cores para cada veículo (uma por rota)
def _gerar_tons_verde(n):
    """N tons de verde interpolados de escuro a claro: (0, 80→240, 0)."""
    if n == 1:
        return [(0, 160, 0)]
    return [(0, int(80 + (240 - 80) * i / (n - 1)), 0) for i in range(n)]

def _gerar_tons_vermelho(n):
    """N tons de vermelho interpolados de escuro a claro: (80→240, 0, 0)."""
    if n == 1:
        return [(180, 0, 0)]
    return [(int(80 + (240 - 80) * i / (n - 1)), 0, 0) for i in range(n)]

# Cores dinâmicas: carros=verdes, motos=vermelhos — funciona com qualquer quantidade
CORES_VEICULOS = _gerar_tons_verde(N_CARROS) + _gerar_tons_vermelho(N_MOTOS)


# Usando benchmark att48
LARGURA, ALTURA = 1500, 800
localizacoes_cidades_att = np.array(att_48_cities_locations)
max_x = max(ponto[0] for ponto in localizacoes_cidades_att)
max_y = max(ponto[1] for ponto in localizacoes_cidades_att)
escala_x = (LARGURA - DESLOCAMENTO_X_GRAFICO - RAIO_NO) / max_x
escala_y = ALTURA / max_y
localizacoes_cidades = [(int(ponto[0] * escala_x + DESLOCAMENTO_X_GRAFICO),
                     int(ponto[1] * escala_y)) for ponto in localizacoes_cidades_att]

# Depósito: seleção conforme DEPOSITO_MODO
if DEPOSITO_MODO == 'central':
    # Calcula o centróide geométrico de todas as cidades
    centroide_x = sum(c[0] for c in localizacoes_cidades) / len(localizacoes_cidades)
    centroide_y = sum(c[1] for c in localizacoes_cidades) / len(localizacoes_cidades)
    # Seleciona a cidade mais próxima ao centróide como depósito
    DEPOSITO = min(localizacoes_cidades,
                   key=lambda c: (c[0] - centroide_x) ** 2 + (c[1] - centroide_y) ** 2)
    print(f"Depósito (central): {DEPOSITO} | Centróide: ({centroide_x:.0f}, {centroide_y:.0f})")
else:  # 'primeiro'
    DEPOSITO = localizacoes_cidades[0]
    print(f"Depósito (primeiro): {DEPOSITO}")

# Cidades a visitar (todas exceto o depósito)
_todas_sem_deposito = [c for c in localizacoes_cidades if c != DEPOSITO]

# Aplica N_CIDADES: amostra aleatória de (N_CIDADES - 1) cidades + o depósito
# Se N_CIDADES >= total disponível, usa todas as cidades
if N_CIDADES < len(_todas_sem_deposito) + 1:
    cidades_sem_deposito = random.sample(_todas_sem_deposito, N_CIDADES - 1)
    print(f"Usando {N_CIDADES} cidades (amostra aleatória de {len(_todas_sem_deposito) + 1} disponíveis)")
else:
    cidades_sem_deposito = _todas_sem_deposito
    print(f"Usando todas as {len(_todas_sem_deposito) + 1} cidades disponíveis")

print(f"Cidades a visitar: {len(cidades_sem_deposito)}")
print(f"Número de veículos: {N_VEICULOS}")
# ----- Fim benchmark att48



# ============================================================================
# ETAPA 1 — CODIFICAÇÃO COMBINATÓRIA (Representação do Cromossomo)
# ============================================================================
# O cromossomo é uma permutação de todas as cidades (sem repetição).
# Para o mTSP, essa permutação é dividida em N sub-rotas (uma por veículo).
# Cada sub-rota parte do depósito, visita suas cidades e retorna ao depósito.
# ============================================================================

def dividir_rota(cromossomo, n_veiculos):
    """Divide um cromossomo em N sub-rotas.
    Se OBJETIVO == 'tempo' ou 'hibrido', usa quebra dinâmica baseada em carga estimada
    para permitir rotas de tamanhos (quantidade de cidades) diferentes."""
    
    if OBJETIVO == 'distancia':
        n_cidades = len(cromossomo)
        tamanho_base = n_cidades // n_veiculos
        resto = n_cidades % n_veiculos

        rotas = []
        inicio = 0
        for i in range(n_veiculos):
            tamanho = tamanho_base + (1 if i < resto else 0)
            rotas.append(cromossomo[inicio:inicio + tamanho])
            inicio += tamanho
        return rotas

    # Divisão dinâmica gulosa considerando a distância para tentar equilibrar o tempo
    # Calcula a carga total aproximada primeiro
    dist_total = distancia_euclidiana(DEPOSITO, cromossomo[0])
    for i in range(len(cromossomo) - 1):
        dist_total += distancia_euclidiana(cromossomo[i], cromossomo[i + 1])
    dist_total += distancia_euclidiana(cromossomo[-1], DEPOSITO)
    
    alvo_dist = dist_total / n_veiculos
    
    rotas = []
    rota_atual = []
    dist_acumulada = 0
    pos_atual = DEPOSITO
    
    for cidade in cromossomo:
        dist_trecho = distancia_euclidiana(pos_atual, cidade)
        
        # Se adicionar a cidade excede muito o alvo e ainda podemos abrir novas rotas
        # E já temos pelo menos uma cidade na rota atual
        if dist_acumulada + dist_trecho > alvo_dist and len(rotas) < n_veiculos - 1 and len(rota_atual) > 0:
            rotas.append(rota_atual)
            rota_atual = [cidade]
            dist_acumulada = distancia_euclidiana(DEPOSITO, cidade)
            pos_atual = cidade
        else:
            rota_atual.append(cidade)
            dist_acumulada += dist_trecho
            pos_atual = cidade
            
    if rota_atual:
        rotas.append(rota_atual)
        
    # Garante que temos exatamente N veículos (veículos ociosos ficam com rotas vazias)
    while len(rotas) < n_veiculos:
        rotas.append([])
        
    return rotas


def calcular_distancia_rota(rota, deposito):
    """Calcula a distância de uma sub-rota: depósito → cidades → depósito."""
    if len(rota) == 0:
        return 0
    dist = distancia_euclidiana(deposito, rota[0])
    for i in range(len(rota) - 1):
        dist += distancia_euclidiana(rota[i], rota[i + 1])
    dist += distancia_euclidiana(rota[-1], deposito)
    return dist


def distancia_euclidiana(a, b):
    """Distância euclidiana entre dois pontos."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


# ============================================================================
# DISTÂNCIA EFETIVA COM REABASTECIMENTO
# ============================================================================
# Quando um veículo não tem autonomia suficiente para chegar à próxima cidade
# E ainda retornar à base depois, ele deve voltar à base, reabastecer, e
# então seguir para a próxima cidade. Isso aumenta a distância real percorrida.
# ============================================================================

def calcular_distancia_efetiva(rota, deposito, autonomia):
    """Calcula distância real incluindo viagens de volta à base para reabastecer.
    Retorna (distância_efetiva, número_de_reabastecimentos)."""
    if not rota:
        return 0, 0

    dist_total = 0
    combustivel = autonomia
    reabastecimentos = 0
    pos_atual = deposito

    for cidade in rota:
        dist_ate_cidade = distancia_euclidiana(pos_atual, cidade)
        
        if REABASTECIMENTO_ATIVO:
            # Preciso chegar na cidade E ter combustível pra voltar à base depois
            dist_cidade_base = distancia_euclidiana(cidade, deposito)
            if combustivel < dist_ate_cidade + dist_cidade_base:
                # Não dá: voltar à base para reabastecer
                dist_volta = distancia_euclidiana(pos_atual, deposito)
                dist_total += dist_volta     # Volta à base
                combustivel = autonomia       # Tanque cheio
                reabastecimentos += 1
                pos_atual = deposito
                dist_ate_cidade = distancia_euclidiana(deposito, cidade)
                
            combustivel -= dist_ate_cidade

        dist_total += dist_ate_cidade
        pos_atual = cidade

    # Volta final à base
    dist_total += distancia_euclidiana(pos_atual, deposito)
    return dist_total, reabastecimentos


def construir_waypoints_reabastecimento(rota, deposito, autonomia):
    """Retorna a sequência real de pontos percorridos, incluindo voltas ao depósito
    para reabastecimento. Útil para desenhar a rota visualmente.
    Retorna lista de (ponto, tipo) onde tipo é 'cidade' ou 'deposito'."""
    if not rota:
        return [(deposito, 'deposito')]

    waypoints = [(deposito, 'deposito')]  # Ponto de partida
    combustivel = autonomia
    pos_atual = deposito

    for cidade in rota:
        dist_ate_cidade = distancia_euclidiana(pos_atual, cidade)
        
        if REABASTECIMENTO_ATIVO:
            dist_cidade_base = distancia_euclidiana(cidade, deposito)
            if combustivel < dist_ate_cidade + dist_cidade_base:
                # Reabastece: volta à base antes de ir à cidade
                waypoints.append((deposito, 'deposito'))  # Volta para base
                combustivel = autonomia
                pos_atual = deposito
                dist_ate_cidade = distancia_euclidiana(deposito, cidade)
                
            combustivel -= dist_ate_cidade

        waypoints.append((cidade, 'cidade'))
        pos_atual = cidade

    waypoints.append((deposito, 'deposito'))  # Volta final
    return waypoints

def calcular_tempo_rota(rota, deposito, veiculo):
    """Calcula o tempo (horas) de uma rota considerando autonomia e velocidade."""
    dist_efetiva, reab = calcular_distancia_efetiva(rota, deposito, veiculo['autonomia'])
    tempo = dist_efetiva / veiculo['velocidade']
    return tempo, dist_efetiva, reab


def calcular_tempos_rotas(rotas, deposito):
    """Calcula tempos, distâncias efetivas e reabastecimentos de todas as rotas.
    Retorna (tempos, dist_efetivas, reabastecimentos) — uma lista por veículo."""
    tempos = []
    dist_efetivas = []
    reabs = []
    for i, rota in enumerate(rotas):
        veiculo = VEICULOS[i]
        tempo, dist_ef, reab = calcular_tempo_rota(rota, deposito, veiculo)
        tempos.append(tempo)
        dist_efetivas.append(dist_ef)
        reabs.append(reab)
    return tempos, dist_efetivas, reabs


# ============================================================================
# ETAPA 2 — FUNÇÃO DE APTIDÃO (FITNESS)
# ============================================================================
# O fitness avalia a DISTÂNCIA EFETIVA percorrida (km), não o tempo.
# Distância efetiva inclui os desvios de reabastecimento à base quando
# o veículo não tem autonomia suficiente para a próxima cidade.
#
# fitness = max(dist_ef) + peso × total_dist_ef
#   - peso=0.0 → só minimiza a rota mais longa (minimax)
#   - peso>0.0 → também penaliza o total percorrido por todos
#
# Esse objetivo incentiva carros (600km autonomia) a cobrirem cidades
# distantes (menos desvios de reabastecimento) e motos (300km) a
# cobrirem cidades próximas — maximizando a eficiência de combustível.
# ============================================================================

def calcular_fitness_mtsp(cromossomo, deposito, n_veiculos, peso_balanceamento=PESO_BALANCEAMENTO):
    """Calcula o fitness do mTSP conforme o OBJETIVO configurado.

    OBJETIVO='distancia' → fitness = max(dist_ef) + peso * total_dist_ef
    OBJETIVO='tempo'     → fitness = max(tempo) + peso * total_tempo
    OBJETIVO='hibrido'   → combina as duas métricas normalizadas por VELOCIDADE_REF:
                           dist_component  (km)
                           tempo_component (horas * VELOCIDADE_REF → km-equivalente)
                           fitness = (1 - ALFA) * dist_component + ALFA * tempo_component
    """
    rotas = dividir_rota(cromossomo, n_veiculos)
    dist_efetivas = []
    tempos = []
    for idx, rota in enumerate(rotas):
        veiculo = VEICULOS[idx]
        dist_ef, _ = calcular_distancia_efetiva(rota, deposito, veiculo['autonomia'])
        tempo, _, _ = calcular_tempo_rota(rota, deposito, veiculo)
        dist_efetivas.append(dist_ef)
        tempos.append(tempo)

    if OBJETIVO == 'tempo':
        amplitude = max(tempos) - min(tempos)
        # O objetivo é minimizar o tempo máximo + forte penalidade no desbalanceamento
        return max(tempos) + (peso_balanceamento * sum(tempos)) + (amplitude * 5.0)

    elif OBJETIVO == 'hibrido':
        # Componente de distância (km)
        dist_component = max(dist_efetivas) + peso_balanceamento * sum(dist_efetivas)
        # Componente de tempo convertido para km-equivalente (h × km/h)
        amplitude_tempo = max(tempos) - min(tempos)
        tempo_component = (max(tempos) + peso_balanceamento * sum(tempos) + amplitude_tempo * 5.0) * VELOCIDADE_REF
        return (1 - ALFA_HIBRIDO) * dist_component + ALFA_HIBRIDO * tempo_component

    else:  # 'distancia'
        return max(dist_efetivas) + peso_balanceamento * sum(dist_efetivas)


def ordenar_populacao_mtsp(populacao, deposito, n_veiculos):
    """Ordena a população pelo fitness mTSP (menor é melhor)."""
    fitness_lista = [calcular_fitness_mtsp(ind, deposito, n_veiculos) for ind in populacao]
    pares = list(zip(populacao, fitness_lista))
    pares.sort(key=lambda x: x[1])
    populacao_ordenada = [p[0] for p in pares]
    fitness_ordenado = [p[1] for p in pares]
    return populacao_ordenada, fitness_ordenado


# ============================================================================
# ETAPA 3 — GERAR POPULAÇÃO INICIAL
# ============================================================================
# A população pode ser gerada de três formas:
#   a) Aleatória — permutações aleatórias das cidades
#   b) Heurística — soluções construídas inteligentemente (melhor ponto de partida)
#      - Vizinho Mais Próximo: começa numa cidade e sempre vai à mais próxima
#      - Convex Hull: constrói rota pelos pontos externos, e insere internos
#   c) Soluções conhecidas — carregar soluções de execuções anteriores
#
# Combinar heurísticas com aleatórias dá diversidade genética + qualidade.
# ============================================================================

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


# ============================================================================
# ETAPA 6 — MUTAÇÃO
# ============================================================================
# A mutação introduz diversidade genética na população, explorando combinações
# que o cruzamento sozinho não alcançaria. Dois aspectos chave:
#   - Probabilidade: com que frequência a mutação acontece (ex: 60%)
#   - Intensidade: quão forte é a alteração no cromossomo
#     · Forte: inverte segmentos, move blocos (inversão, or-opt)
#     · Fraca: troca apenas dois genes adjacentes (swap)
#
# Para mTSP, temos 4 operadores especializados:
#   1. Inversão de segmento (30%) — reverte trecho do cromossomo (2-opt move)
#   2. Troca entre rotas (30%) — swap entre rota pesada ↔ rota leve
#   3. Or-opt (25%) — move 1-3 cidades consecutivas para outra posição
#   4. Swap adjacente (15%) — mutação clássica simples
# ============================================================================

# --- Operadores de Mutação Avançados para mTSP ---
def mutacao_inversao(cromossomo):
    """Inverte um segmento aleatório do cromossomo (movimento 2-opt)."""
    n = len(cromossomo)
    if n < 3:
        return cromossomo
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, min(i + n // 2, n - 1))  # Segmentos até n//2
    novo = list(cromossomo)
    novo[i:j+1] = reversed(novo[i:j+1])
    return novo


def mutacao_troca_entre_rotas(cromossomo, n_veiculos, deposito):
    """TROCA (swap) uma cidade da rota pesada com uma da rota leve.
    Mantém a contagem de cidades igual para preservar a divisão."""
    rotas = dividir_rota(cromossomo, n_veiculos)
    distancias = [calcular_distancia_rota(rota, deposito) for rota in rotas]

    idx_max = distancias.index(max(distancias))
    idx_min = distancias.index(min(distancias))

    if idx_max == idx_min or len(rotas[idx_max]) <= 1 or len(rotas[idx_min]) <= 1:
        return cromossomo

    rota_pesada = list(rotas[idx_max])
    rota_leve = list(rotas[idx_min])

    # Testa todas as trocas possíveis e escolhe a melhor
    melhor_melhoria = 0
    melhor_i = 0
    melhor_j = 0
    dist_pesada_atual = distancias[idx_max]
    dist_leve_atual = distancias[idx_min]
    max_atual = max(dist_pesada_atual, dist_leve_atual)

    for i in range(len(rota_pesada)):
        for j in range(len(rota_leve)):
            # Simula a troca
            rota_p_teste = rota_pesada[:i] + [rota_leve[j]] + rota_pesada[i+1:]
            rota_l_teste = rota_leve[:j] + [rota_pesada[i]] + rota_leve[j+1:]
            dist_p = calcular_distancia_rota(rota_p_teste, deposito)
            dist_l = calcular_distancia_rota(rota_l_teste, deposito)
            max_novo = max(dist_p, dist_l)
            melhoria = max_atual - max_novo
            if melhoria > melhor_melhoria:
                melhor_melhoria = melhoria
                melhor_i = i
                melhor_j = j

    if melhor_melhoria <= 0:
        return cromossomo

    # Aplica a melhor troca no cromossomo original
    novo = list(cromossomo)
    # Calcula as posições absolutas no cromossomo
    inicio_pesada = 0
    for k in range(idx_max):
        inicio_pesada += len(rotas[k])
    inicio_leve = 0
    for k in range(idx_min):
        inicio_leve += len(rotas[k])

    pos_pesada = inicio_pesada + melhor_i
    pos_leve = inicio_leve + melhor_j
    novo[pos_pesada], novo[pos_leve] = novo[pos_leve], novo[pos_pesada]
    return novo


def mutacao_or_opt(cromossomo):
    """Move 1-3 cidades consecutivas para outra posição."""
    n = len(cromossomo)
    if n < 4:
        return cromossomo
    novo = list(cromossomo)
    tamanho_seg = random.randint(1, min(3, n // 2))
    i = random.randint(0, n - tamanho_seg)
    segmento = novo[i:i + tamanho_seg]
    del novo[i:i + tamanho_seg]
    j = random.randint(0, len(novo))
    for k, cidade in enumerate(segmento):
        novo.insert(j + k, cidade)
    return novo


def mutacao_mtsp(cromossomo, probabilidade_mutacao, n_veiculos, deposito):
    """Aplica um operador de mutação aleatório com a probabilidade dada."""
    if random.random() >= probabilidade_mutacao:
        return cromossomo

    operador = random.random()
    if operador < 0.3:
        return mutacao_inversao(cromossomo)
    elif operador < 0.6:
        return mutacao_troca_entre_rotas(cromossomo, n_veiculos, deposito)
    elif operador < 0.85:
        return mutacao_or_opt(cromossomo)
    else:
        # Mutação simples original (swap adjacente)
        return mutate(cromossomo, 1.0)


# --- Refinamento 2-opt ---
def dois_opt(rota, deposito, autonomia):
    """Aplica 2-opt numa sub-rota usando distância EFETIVA (com reabastecimento)."""
    if len(rota) < 3:
        return rota
    melhorou = True
    melhor_rota = list(rota)
    melhor_dist, _ = calcular_distancia_efetiva(melhor_rota, deposito, autonomia)
    while melhorou:
        melhorou = False
        for i in range(len(melhor_rota) - 1):
            for j in range(i + 2, len(melhor_rota)):
                nova_rota = melhor_rota[:i] + melhor_rota[i:j+1][::-1] + melhor_rota[j+1:]
                nova_dist, _ = calcular_distancia_efetiva(nova_rota, deposito, autonomia)
                if nova_dist < melhor_dist:
                    melhor_rota = nova_rota
                    melhor_dist = nova_dist
                    melhorou = True
    return melhor_rota


def aplicar_2opt_mtsp(cromossomo, deposito, n_veiculos):
    """Aplica 2-opt em cada sub-rota do cromossomo usando a autonomia do veículo."""
    rotas = dividir_rota(cromossomo, n_veiculos)
    rotas_otimizadas = []
    for idx, rota in enumerate(rotas):
        veiculo = VEICULOS[idx]
        rotas_otimizadas.append(dois_opt(rota, deposito, veiculo['autonomia']))
    novo_cromossomo = []
    for rota in rotas_otimizadas:
        novo_cromossomo.extend(rota)
    return novo_cromossomo
# --- Fim 2-opt ---


# --- Heurística do Vizinho Mais Próximo (Greedy / Gulosa) ---
# Estratégia Gulosa: a cada passo, escolhe a cidade não visitada mais próxima.
# Vantagem: rápida e gera soluções razoáveis. Desvantagem: pode ficar presa em ótimos locais.
def vizinho_mais_proximo(cidades, indice_inicial=0):
    """Gera uma rota usando a heurística de vizinho mais próximo."""
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


# --- Geração Aleatória (Diversidade Genética) ---
# Gera permutações totalmente aleatórias para garantir diversidade na população.
def gerar_populacao_aleatoria_mtsp(cidades, tamanho_populacao):
    """Gera população aleatória de permutações das cidades (sem o depósito)."""
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = list(cidades)
        random.shuffle(individuo)
        populacao.append(individuo)
    return populacao


# --- EXECUÇÃO: Criar População Inicial baseada na HEURÍSTICA escolhida ---
# Mistura soluções heurísticas (qualidade) + aleatórias (diversidade)
if HEURISTICA == 1:
    # Vizinho Mais Próximo — gera várias soluções começando de cidades diferentes
    print("Heurística selecionada: Vizinho Mais Próximo")
    solucoes_vmp = []
    for i in range(min(len(cidades_sem_deposito), TAMANHO_POPULACAO)):
        solucoes_vmp.append(vizinho_mais_proximo(cidades_sem_deposito, indice_inicial=i))
    restante = TAMANHO_POPULACAO - len(solucoes_vmp)
    solucoes_aleatorias = gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, restante) if restante > 0 else []
    populacao = solucoes_vmp + solucoes_aleatorias

elif HEURISTICA == 2:
    # Convex Hull — 1 solução heurística + (N-1) aleatórias
    print("Heurística selecionada: Convex Hull")
    solucao_envoltoria = insercao_envoltoria_convexa(cidades_sem_deposito)
    populacao = [solucao_envoltoria] + gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, TAMANHO_POPULACAO - 1)

elif HEURISTICA == 3:
    # Totalmente aleatória — máxima diversidade genética, sem viés heurístico
    print("Heurística selecionada: Aleatório")
    populacao = gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, TAMANHO_POPULACAO)

else:
    print("ERRO: Heurística inválida! Use 1 (Vizinho Mais Próximo), 2 (Convex Hull) ou 3 (Aleatório).")
    pygame.quit()
    sys.exit(1)

fitness_inicial = calcular_fitness_mtsp(populacao[0], DEPOSITO, N_VEICULOS)
print(f"Fitness da melhor solução inicial (mTSP): {round(fitness_inicial, 2)}")
# --- Fim da Geração da População Inicial ---


# Inicializar Pygame
pygame.init()
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption(f"mTSP - {N_VEICULOS} Veículos usando Pygame")
relogio = pygame.time.Clock()
contador_geracoes = itertools.count(start=1)  # Iniciar o contador em 1

melhores_fitness = []
melhor_fitness_global = float('inf')
melhor_solucao_global = None  # Guardamos apenas o melhor global (nunca regride)
geracoes_sem_melhora = 0


# ============================================================================
# LOOP EVOLUTIVO PRINCIPAL DO ALGORITMO GENÉTICO
# ============================================================================
# A cada geração, o AG executa o ciclo completo:
#   1. Avaliar Aptidão (Fitness) — calcula quão boa é cada solução
#   2. Validar Condição de Término — verifica convergência ou nº máx gerações
#   3. Seleção — escolhe os pais com base na aptidão (roleta proporcional)
#   4. Cruzamento (Crossover) — combina pais para gerar filhos (OX1)
#   5. Mutação — introduz variações aleatórias para explorar novas soluções
#   6. Substituição da População — substitui a geração anterior pela nova
# ============================================================================
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

    # ------------------------------------------------------------------
    # ETAPA 2 (no loop) — AVALIAR APTIDÃO (FITNESS)
    # ------------------------------------------------------------------
    # Calcula o fitness de todos os cromossomos e ordena do melhor ao pior.
    # Função de Fitness: somar distâncias de todas as sub-rotas + penalidade
    # de balanceamento (minimax). Menor fitness = melhor solução.
    # ------------------------------------------------------------------
    populacao, fitness_populacao = ordenar_populacao_mtsp(populacao, DEPOSITO, N_VEICULOS)

    melhor_fitness = fitness_populacao[0]
    melhor_solucao = populacao[0]

    melhores_fitness.append(melhor_fitness)

    # Desenhar gráfico de convergência
    if OBJETIVO == 'distancia':
        label_fitness = "Distância Efetiva (km)"
    elif OBJETIVO == 'tempo':
        label_fitness = "Tempo (h)"
    else:
        label_fitness = f"Híbrido (α={ALFA_HIBRIDO})"
    draw_plot(tela, list(range(len(melhores_fitness))),
              melhores_fitness, y_label=f"Fitness - {label_fitness}")

    # Desenhar cidades e depósito
    draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
    # Depósito com cor especial e tamanho maior
    pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
    pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)

    # Desenhar rotas da melhor solução (uma cor por veículo)
    rotas_melhor = dividir_rota(melhor_solucao, N_VEICULOS)
    for idx, rota in enumerate(rotas_melhor):
        cor = CORES_VEICULOS[idx % len(CORES_VEICULOS)]
        veiculo = VEICULOS[idx]

        if len(rota) == 0:
            continue
        if len(rota) == 1:
            pygame.draw.line(tela, cor, DEPOSITO, rota[0], 3)
            pygame.draw.line(tela, cor, rota[0], DEPOSITO, 3)
            continue

        # Construir waypoints reais (com paradas de reabastecimento)
        waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, veiculo['autonomia'])
        pontos = [wp[0] for wp in waypoints]
        tipos = [wp[1] for wp in waypoints]

        # Desenhar segmento a segmento
        for i in range(len(pontos) - 1):
            p1, p2 = pontos[i], pontos[i + 1]
            # Segmento de reabastecimento: tracejado e mais fino
            if tipos[i] == 'deposito' or tipos[i + 1] == 'deposito':
                # Linha tracejada: alternando segmentos de 8px
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                comprimento = max(1, int((dx**2 + dy**2)**0.5))
                for t in range(0, comprimento, 14):
                    frac1 = t / comprimento
                    frac2 = min((t + 7) / comprimento, 1.0)
                    x1 = int(p1[0] + dx * frac1)
                    y1 = int(p1[1] + dy * frac1)
                    x2 = int(p1[0] + dx * frac2)
                    y2 = int(p1[1] + dy * frac2)
                    pygame.draw.line(tela, cor, (x1, y1), (x2, y2), 2)
                # Pequeno símbolo de bomba no meio do segmento de retorno
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                pygame.draw.circle(tela, cor, (mx, my), 5)
                pygame.draw.circle(tela, PRETO, (mx, my), 5, 1)
            else:
                pygame.draw.line(tela, cor, p1, p2, 3)

    # Desenhar rota do segundo melhor (cinza, para referência)
    if len(populacao) > 1:
        rotas_segundo = dividir_rota(populacao[1], N_VEICULOS)
        for rota in rotas_segundo:
            if len(rota) >= 2:
                rota_completa = [DEPOSITO] + list(rota) + [DEPOSITO]
                draw_paths(tela, rota_completa, (180, 180, 180), width=1)

    print(f"Geração {geracao}: Melhor fitness = {round(melhor_fitness, 2)}")

    # ------------------------------------------------------------------
    # ETAPA 3 — VALIDAR CONDIÇÃO DE TÉRMINO DO FLUXO
    # ------------------------------------------------------------------
    # Critérios implementados:
    #   a) Convergência: para se não melhora após N gerações consecutivas
    #   b) Alcance da solução ótima: se fitness ≤ threshold (não usado aqui)
    # O critério principal usado é a CONVERGÊNCIA (estagnação).
    # ------------------------------------------------------------------
    if melhor_fitness < melhor_fitness_global:
        # Nova melhor solução global encontrada
        melhor_fitness_global = melhor_fitness
        melhor_solucao_global = melhor_solucao[:]  # Cópia da melhor solução
        geracoes_sem_melhora = 0
    else:
        geracoes_sem_melhora += 1

    # Parada por convergência: N gerações sem melhora
    if geracoes_sem_melhora >= GERACOES_SEM_MELHORA_PARA_PARAR:
        print(f"\nParada automática: {GERACOES_SEM_MELHORA_PARA_PARAR} gerações sem melhora.")
        executando = False

    # ------------------------------------------------------------------
    # ETAPA 7 — SUBSTITUIÇÃO DA POPULAÇÃO (Elitismo)
    # ------------------------------------------------------------------
    # Elitismo: os 3 melhores indivíduos passam direto para a próxima
    # geração, garantindo que boas soluções nunca se percam.
    # O restante da população é preenchido por novos filhos.
    # ------------------------------------------------------------------
    # Elitismo: os 3 melhores passam direto (sem modificação)
    # Nota: 2-opt periódico foi removido — ele otimiza por distância pura,
    # ignorando reabastecimento, causando regressão no fitness baseado em tempo.
    nova_populacao = populacao[:3]

    # --- Injeção de imigrantes aleatórios a cada 50 gerações ---
    # Preserva diversidade genética e evita convergência prematura
    N_IMIGRANTES = max(5, TAMANHO_POPULACAO // 10)
    if geracoes_sem_melhora > 0 and geracoes_sem_melhora % 50 == 0:
        imigrantes = gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, N_IMIGRANTES)
        nova_populacao.extend(imigrantes)

    # Seleção por Torneio (k=3)
    def torneio(pop, fit, k=3):
        indices = random.sample(range(len(pop)), k)
        melhor_idx = min(indices, key=lambda i: fit[i])
        return pop[melhor_idx]

    while len(nova_populacao) < TAMANHO_POPULACAO:

        # --------------------------------------------------------------
        # ETAPA 4 — SELEÇÃO DOS PAIS
        # --------------------------------------------------------------
        # Método: Roleta / Proporcional ao Fitness
        # Indivíduos com menor distância (melhor fitness) têm MAIOR
        # probabilidade de serem selecionados como pais.
        # Probabilidade = 1/fitness (inversamente proporcional).
        # Outros métodos possíveis: Torneio, Ranking.
        # --------------------------------------------------------------
        pai1 = torneio(populacao, fitness_populacao)
        pai2 = torneio(populacao, fitness_populacao)

        # --------------------------------------------------------------
        # ETAPA 5 — CRUZAMENTO (CROSSOVER)
        # --------------------------------------------------------------

        filho1 = order_crossover(pai1, pai2)
        filho2 = order_crossover(pai2, pai1)

        # --------------------------------------------------------------
        # ETAPA 6 — MUTAÇÃO
        # --------------------------------------------------------------

        filho1 = mutacao_mtsp(filho1, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)
        filho2 = mutacao_mtsp(filho2, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)

        nova_populacao.append(filho1)
        nova_populacao.append(filho2)

    # ------------------------------------------------------------------
    # Substituição: a nova geração substitui a anterior por completo
    # (exceto os 3 elitistas que foram preservados no início).
    # ------------------------------------------------------------------
    populacao = nova_populacao

    pygame.display.flip()
    relogio.tick(FPS)

# ============================================================================
# CICLO ITERATIVO DE REFINAMENTO: AG → 2-opt → AG → 2-opt → ...
# ============================================================================
melhor_solucao_final = melhor_solucao_global

ciclo = 0
GERACOES_REFINAMENTO = 400
fitness_antes_ciclo = calcular_fitness_mtsp(melhor_solucao_final, DEPOSITO, N_VEICULOS)

while True:
    ciclo += 1
    print(f"\n--- Ciclo de refinamento {ciclo} ---")
    print(f"  Fitness antes do 2-opt: {round(fitness_antes_ciclo, 2)}")
    melhor_solucao_final = aplicar_2opt_mtsp(melhor_solucao_final, DEPOSITO, N_VEICULOS)
    fitness_pos_2opt = calcular_fitness_mtsp(melhor_solucao_final, DEPOSITO, N_VEICULOS)
    print(f"  Fitness após 2-opt: {round(fitness_pos_2opt, 2)}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            pygame.quit()
            sys.exit()
    tela.fill(BRANCO)
    draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
    pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
    pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)
    rotas_temp = dividir_rota(melhor_solucao_final, N_VEICULOS)
    for idx_rota, rota_t in enumerate(rotas_temp):
        cor_veiculo = CORES_VEICULOS[idx_rota % len(CORES_VEICULOS)]
        autonomia_v = VEICULOS[idx_rota]['autonomia']
        waypoints = construir_waypoints_reabastecimento(rota_t, DEPOSITO, autonomia_v)
        for p_idx in range(len(waypoints) - 1):
            inicio_p, t_inicio = waypoints[p_idx]
            fim_p, t_fim = waypoints[p_idx + 1]
            if inicio_p == DEPOSITO or fim_p == DEPOSITO:
                pygame.draw.line(tela, cor_veiculo, inicio_p, fim_p, 1)
                mid_x = (inicio_p[0] + fim_p[0]) // 2
                mid_y = (inicio_p[1] + fim_p[1]) // 2
                pygame.draw.circle(tela, cor_veiculo, (mid_x, mid_y), 3)
            else:
                pygame.draw.line(tela, cor_veiculo, inicio_p, fim_p, 3)
    draw_plot(tela, list(range(len(melhores_fitness))), melhores_fitness, x_label='Geração', y_label='[REFINAMENTO 2-OPT]')
    draw_info_overlay(tela, f"{geracao} (+{ciclo} 2-opt)", fitness_pos_2opt, "?", N_VEICULOS)
    pygame.display.flip()

    pop_refinamento = [melhor_solucao_final[:]]
    for _ in range(TAMANHO_POPULACAO // 2):
        variante = mutacao_mtsp(melhor_solucao_final[:], 1.0, N_VEICULOS, DEPOSITO)
        pop_refinamento.append(variante)
    restante_ref = TAMANHO_POPULACAO - len(pop_refinamento)
    pop_refinamento += gerar_populacao_aleatoria_mtsp(cidades_sem_deposito, restante_ref)
    sem_melhora_ref = 0
    melhor_fitness_ref = fitness_pos_2opt
    melhor_sol_ref = melhor_solucao_final[:]
    for gen_ref in range(1, GERACOES_REFINAMENTO * 3 + 1):
        pop_refinamento, fit_ref = ordenar_populacao_mtsp(pop_refinamento, DEPOSITO, N_VEICULOS)
        if fit_ref[0] < melhor_fitness_ref:
            melhor_fitness_ref = fit_ref[0]
            melhor_sol_ref = pop_refinamento[0][:]
            sem_melhora_ref = 0
        else:
            sem_melhora_ref += 1
        if sem_melhora_ref >= GERACOES_REFINAMENTO:
            print(f"  AG extra parou na geração {gen_ref} ({GERACOES_REFINAMENTO} sem melhora)")
            break
        nova_pop_ref = pop_refinamento[:3]
        while len(nova_pop_ref) < TAMANHO_POPULACAO:
            def torneio_ref(pop, fit, k=3):
                indices = random.sample(range(len(pop)), k)
                melhor_idx = min(indices, key=lambda i: fit[i])
                return pop[melhor_idx]
            p1 = torneio_ref(pop_refinamento, fit_ref)
            p2 = torneio_ref(pop_refinamento, fit_ref)
            f1 = order_crossover(p1, p2)
            f2 = order_crossover(p2, p1)
            f1 = mutacao_mtsp(f1, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)
            f2 = mutacao_mtsp(f2, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)
            nova_pop_ref.append(f1)
            nova_pop_ref.append(f2)
        pop_refinamento = nova_pop_ref
    print(f"  Fitness após AG extra: {round(melhor_fitness_ref, 2)}")
    if melhor_fitness_ref < fitness_antes_ciclo - 0.01:
        melhoria = fitness_antes_ciclo - melhor_fitness_ref
        print(f"  ✓ Melhoria de {round(melhoria, 2)} km! Continuando...")
        melhor_solucao_final = melhor_sol_ref
        fitness_antes_ciclo = melhor_fitness_ref
    else:
        print(f"  ✗ Sem melhoria. Encerrando refinamento.")
        break
print(f"\nRefinamento concluído: {ciclo} ciclo(s), fitness final = {round(fitness_antes_ciclo, 2)}")

rotas_finais = dividir_rota(melhor_solucao_final, N_VEICULOS)
rotas_finais = [list(rota) for rota in rotas_finais]
distancias_final = [calcular_distancia_rota(r, DEPOSITO) for r in rotas_finais]
tempos_final, dist_ef_final, reab_final = calcular_tempos_rotas(rotas_finais, DEPOSITO)
melhor_fitness_final = sum(dist_ef_final)

# Salvar o melhor indivíduo em arquivo
os.makedirs("rotas", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ARQUIVO_MELHOR_SOLUCAO = f"rotas/melhor_solucao_mtsp_{timestamp}.json"

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
        'timestamp': timestamp,
        'fitness': melhor_fitness_final,
        'n_veiculos': N_VEICULOS,
        'deposito': DEPOSITO,
        'rotas': [list(rota) for rota in rotas_finais],
        'cidades_por_rota': [len(rota) for rota in rotas_finais],
        'distancias_por_rota': [round(calcular_distancia_rota(r, DEPOSITO), 2) for r in rotas_finais],
        'numero_geracoes': geracao
    }
    with open(ARQUIVO_MELHOR_SOLUCAO, 'w') as f:
        json.dump(dados, f, indent=2)
    print(f"Nova melhor solução mTSP salva! Fitness: {round(melhor_fitness_final, 2)} em {geracao} gerações.")


# ============================================================================
# ETAPA 9 — RESULTADOS E PERSISTÊNCIA
# ============================================================================
# Usar diretamente o melhor resultado do AG (sem pós-processamento).
# Salvar em JSON, capturar screenshot e gerar relatório via agente ChatGPT.
# ============================================================================

# === Capturar screenshot do resultado final ===
# Redesenhar a tela com as rotas rebalanceadas
tela.fill(BRANCO)
draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)

for idx, rota in enumerate(rotas_finais):
    if len(rota) == 0:
        continue
    cor = CORES_VEICULOS[idx % len(CORES_VEICULOS)]
    autonomia_v = VEICULOS[idx]['autonomia']
    waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, autonomia_v)

    for p_idx in range(len(waypoints) - 1):
        inicio_p, t_inicio = waypoints[p_idx]
        fim_p, t_fim = waypoints[p_idx + 1]
        if inicio_p == DEPOSITO or fim_p == DEPOSITO:
            # Trecho de reabastecimento: linha fina + marcador no ponto médio
            pygame.draw.line(tela, cor, inicio_p, fim_p, 1)
            mid_x = (inicio_p[0] + fim_p[0]) // 2
            mid_y = (inicio_p[1] + fim_p[1]) // 2
            pygame.draw.circle(tela, cor, (mid_x, mid_y), 3)
        else:
            # Trecho normal entre cidades: linha grossa
            pygame.draw.line(tela, cor, inicio_p, fim_p, 3)

# Adicionar texto com informações na tela
fonte = pygame.font.SysFont('Arial', 14)
for idx, rota in enumerate(rotas_finais):
    v = VEICULOS[idx]
    cor = CORES_VEICULOS[idx % len(CORES_VEICULOS)]
    texto = fonte.render(f"{v['tipo']} V{idx+1}: {len(rota)} cid, {round(dist_ef_final[idx],1)}km, {round(tempos_final[idx],2)}h, {reab_final[idx]}⛽", True, cor)
    tela.blit(texto, (10, 10 + idx * 20))
texto_total = fonte.render(f"Total: {round(melhor_fitness_final, 2)}km | Dif tempo: {round(max(tempos_final) - min(tempos_final), 2)}h", True, PRETO)
tela.blit(texto_total, (10, 10 + N_VEICULOS * 20))

pygame.display.flip()

# Salvar screenshot
nome_screenshot = f"rotas/resultado_mtsp_{timestamp}.png"
pygame.image.save(tela, nome_screenshot)
print(f"Screenshot salvo: {nome_screenshot}")

# Imprimir resumo das rotas
print(f"\n=== Resumo mTSP ({N_VEICULOS} veículos: {N_CARROS} carros + {N_MOTOS} motos) ===")
tempos_resumo, dist_ef_resumo, reab_resumo = calcular_tempos_rotas(rotas_finais, DEPOSITO)
for idx, rota in enumerate(rotas_finais):
    v = VEICULOS[idx]
    print(f"{v['tipo']} V{idx+1}: {len(rota)} cidades, dist_efetiva={round(dist_ef_resumo[idx], 2)}km, "
          f"tempo={round(tempos_resumo[idx], 2)}h, reabastecimentos={reab_resumo[idx]}")
distancia_total_real = sum(dist_ef_resumo)
tempo_total = sum(tempos_resumo)
media_tempo = tempo_total / len(tempos_resumo)
desvio_final = (sum((t - media_tempo)**2 for t in tempos_resumo) / len(tempos_resumo)) ** 0.5
print(f"Distância efetiva total: {round(distancia_total_real, 2)}km")
print(f"Tempo total: {round(tempo_total, 2)}h")
print(f"Desvio padrão entre tempos: {round(desvio_final, 2)}h")
print(f"Balanço tempo: {round(min(tempos_resumo), 2)}h - {round(max(tempos_resumo), 2)}h (diferença: {round(max(tempos_resumo) - min(tempos_resumo), 2)}h)")
print(f"Total reabastecimentos: {sum(reab_resumo)}")

if RELATORIO_GPT:
    # === Agente OpenAI - Análise das Rotas ===
    print("\n=== Enviando resultados para o agente ChatGPT... ===\n")

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada no arquivo .env")
    else:
        client = OpenAI(api_key=api_key)

        # Montar dados das rotas com coordenadas originais do benchmark ATT48
        descricao_rotas = ""
        for idx, rota in enumerate(rotas_finais):
            v = VEICULOS[idx]
            descricao_rotas += f"\n### {v['tipo']} Veículo {idx + 1} ({len(rota)} cidades, "
            descricao_rotas += f"dist_efetiva: {round(dist_ef_final[idx], 2)}km, "
            descricao_rotas += f"tempo: {round(tempos_final[idx], 2)}h, "
            descricao_rotas += f"reabastecimentos: {reab_final[idx]})\n"
            descricao_rotas += f"Trajeto: Depósito → "
            cidades_rota = []
            for cidade in rota:
                # Encontrar o índice original da cidade no benchmark
                idx_cidade = localizacoes_cidades.index(cidade)
                coord_original = att_48_cities_locations[idx_cidade]
                cidades_rota.append(f"Cidade {idx_cidade + 1} ({coord_original[0]}, {coord_original[1]})")
            descricao_rotas += " → ".join(cidades_rota)
            descricao_rotas += " → Depósito\n"

        prompt = f"""Você é um analista de logística especializado em otimização de rotas entre hospitais.
    Analise os resultados do problema mTSP (Multiple Travelling Salesman Problem) abaixo.

    ## Dados do Problema
    - Benchmark: ATT48 (48 hospitais)
    - Número de veículos: {N_VEICULOS} ({N_CARROS} carros + {N_MOTOS} motos)
    - Depósito (ponto de partida/chegada): ({att_48_cities_locations[0][0]}, {att_48_cities_locations[0][1]})
    - Algoritmo: Genético com heurística {'Vizinho Mais Próximo' if HEURISTICA == 1 else 'Convex Hull'}
    - Objetivo: {OBJETIVO}
    - Gerações: {geracao}
    - Fitness final: {round(melhor_fitness_final, 2)}

    ## Resultados das Rotas
    {descricao_rotas}

    ## Estatísticas
    - Distância efetiva total: {round(distancia_total_real, 2)}km
    - Tempo total: {round(tempo_total, 2)}h
    - Desvio padrão entre tempos: {round(desvio_final, 2)}h
    - Diferença entre maior e menor tempo: {round(max(tempos_final) - min(tempos_final), 2)}h
    - Total de reabastecimentos: {sum(reab_final)}

    ## Instruções
    1. Descreva o trajeto de cada veículo de forma clara e objetiva
    2. Analise o balanceamento das rotas (distâncias e tempos dos veículos estão equilibrados?)
    3. Dê uma nota de 1 a 10 para o balanceamento
    4. Sugira possíveis melhorias

    Responda em português brasileiro."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um analista de logística especializado em otimização de rotas de veículos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            resposta = response.choices[0].message.content
            print("=" * 60)
            print("ANÁLISE DO AGENTE ChatGPT")
            print("=" * 60)
            print(resposta)
            print("=" * 60)
        except Exception as e:
            print(f"Erro ao chamar a API OpenAI: {e}")

else:
    print("Relatório GPT desativado.")

# Encerrar programa
pygame.quit()
sys.exit()
