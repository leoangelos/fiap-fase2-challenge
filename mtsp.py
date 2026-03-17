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
RELATORIO_GPT = True

# Parâmetros do Algoritmo Genético
N_CIDADES = 48                          # Número de cidades Max 48
TAMANHO_POPULACAO = 100                 # Tamanho da população de cromossomos

PROBABILIDADE_MUTACAO = 0.8             # Probabilidade de mutação (intensidade: alta = explora mais)
GERACOES_SEM_MELHORA_PARA_PARAR = 800   # Critério de convergência: para após N gerações sem melhora
HEURISTICA = 3                          # 1 = Vizinho Mais Próximo | 2 = Convex Hull | 3 = Aleatório
PESO_BALANCEAMENTO = 0.6                # Penalidade por desbalanceamento entre rotas

# Priorização de Hospitais
# Quando ativa, hospitais com prioridade alta são penalizados se aparecem
# tarde na rota (longe do depósito na sequência de visitas).
PRIORIDADE_ATIVA = True                  # True = ativa penalidade por prioridade
PESO_PRIORIDADE = 50.0                   # Peso da penalidade de prioridade no fitness
# Pesos por nível: quanto maior o peso, mais o AG prioriza visitar cedo na rota.
# Hospitais urgentes (prioridade 0) têm peso alto → AG é forçado a visitá-los primeiro.
PESOS_NIVEL_PRIORIDADE = {
    0: 3.0,   # Alta  — urgência máxima: AG prioriza visitar no início da rota
    1: 1.0,   # Média — urgência moderada
    2: 0.0,   # Baixa — sem urgência, pode ir em qualquer posição
}

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
N_CARROS = 4                            # Quantidade de carros
N_MOTOS = 2                             # Quantidade de motos
N_VEICULOS = N_CARROS + N_MOTOS         # Total de veículos

VELOCIDADE_CARRO = 100                   # km/h
VELOCIDADE_MOTO = 120                   # km/h
AUTONOMIA_CARRO = 1200                   # km (precisa reabastecer na base)
AUTONOMIA_MOTO = 400                    # km (precisa reabastecer na base)

# Configuração de Operação
REABASTECIMENTO_ATIVO = True             # Se False, veículos ganham autonomia infinita
CAPACIDADE_CARGA = True                  # Se True, veículos têm limite de cidades por viagem

# Capacidade de carga (máximo de cidades visitadas antes de voltar ao depósito)
CAPACIDADE_CARRO = 6                     # Carro: 6 cidades por viagem
CAPACIDADE_MOTO = 2                      # Moto: 2 cidades por viagem

# Lista de veículos: primeiros N_CARROS são carros, restante são motos
VEICULOS = (
    [{'tipo': 'Carro 🚗', 'velocidade': VELOCIDADE_CARRO, 'autonomia': AUTONOMIA_CARRO, 'capacidade': CAPACIDADE_CARRO}] * N_CARROS +
    [{'tipo': 'Moto 🏍️', 'velocidade': VELOCIDADE_MOTO, 'autonomia': AUTONOMIA_MOTO, 'capacidade': CAPACIDADE_MOTO}] * N_MOTOS
)

# Definição de cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
AZUL = (0, 100, 255)                     # Cor dos postos de gasolina

AMARELO = (200, 200, 0)

# Postos de gasolina — importados do benchmark_att48.py
# O depósito NÃO é posto de gasolina
POSTOS_GASOLINA_ATT = att_48_postos_gasolina

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

# Postos de gasolina: escalar para coordenadas de tela (mesma transformação das cidades)
POSTOS_GASOLINA = [(int(p[0] * escala_x + DESLOCAMENTO_X_GRAFICO),
                    int(p[1] * escala_y)) for p in POSTOS_GASOLINA_ATT]
if REABASTECIMENTO_ATIVO:
    print(f"Reabastecimento ativo: {len(POSTOS_GASOLINA)} postos de gasolina no mapa")
    for _ip, _pp in enumerate(POSTOS_GASOLINA):
        print(f"  Posto {_ip+1}: {_pp}")
else:
    print("Reabastecimento desativado (autonomia infinita).")

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

# Construir mapa de prioridade: coordenada escalada → (prioridade, nome_hospital)
# Usado pela função de fitness quando PRIORIDADE_ATIVA = True
mapa_prioridade = {}
for _i, _coord in enumerate(localizacoes_cidades):
    mapa_prioridade[_coord] = {
        'prioridade': att_48_priorities[_i],
        'nome': att_48_hospitals[_i],
    }

if PRIORIDADE_ATIVA:
    _labels = {0: 'Alta', 1: 'Média', 2: 'Baixa'}
    _contagem = {0: 0, 1: 0, 2: 0}
    for _c in cidades_sem_deposito:
        _contagem[mapa_prioridade[_c]['prioridade']] += 1
    print(f"Prioridades ativas: Alta={_contagem[0]}, Média={_contagem[1]}, Baixa={_contagem[2]}")
else:
    print("Prioridades desativadas.")

if CAPACIDADE_CARGA:
    print(f"Capacidade de carga ativa: Carro={CAPACIDADE_CARRO} cidades, Moto={CAPACIDADE_MOTO} cidades")
else:
    print("Capacidade de carga desativada.")
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
# DISTÂNCIA EFETIVA COM REABASTECIMENTO EM POSTOS DE GASOLINA
# ============================================================================
# Quando REABASTECIMENTO_ATIVO=True, o veículo deve ir ao posto de gasolina
# mais próximo antes que sua autonomia acabe. O depósito NÃO é posto.
# Quando CAPACIDADE_CARGA=True, o veículo volta ao depósito após visitar
# N cidades (para buscar mais itens). Ao voltar ao depósito, também reabastece.
# ============================================================================

def posto_mais_proximo(posicao, postos):
    """Retorna o posto de gasolina mais próximo da posição dada."""
    return min(postos, key=lambda p: distancia_euclidiana(posicao, p))


def calcular_distancia_efetiva(rota, deposito, autonomia, capacidade=None):
    """Calcula distância real incluindo desvios para postos de gasolina
    (reabastecimento) e/ou retornos ao depósito (recarga de carga).
    Retorna (distância_efetiva, número_de_paradas_extras)."""
    if not rota:
        return 0, 0

    dist_total = 0
    combustivel = autonomia
    cidades_visitadas = 0
    paradas = 0
    pos_atual = deposito

    for cidade in rota:
        dist_ate_cidade = distancia_euclidiana(pos_atual, cidade)
        precisa_carga = False
        precisa_combustivel = False

        # Verificar restrição de capacidade de carga → volta ao depósito
        if CAPACIDADE_CARGA and capacidade is not None:
            if cidades_visitadas >= capacidade:
                precisa_carga = True

        # Verificar restrição de combustível → vai ao posto mais próximo
        if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA:
            # Preciso chegar na cidade E da cidade alcançar o posto mais próximo
            posto_prox = posto_mais_proximo(cidade, POSTOS_GASOLINA)
            dist_cidade_posto = distancia_euclidiana(cidade, posto_prox)
            if combustivel < dist_ate_cidade + dist_cidade_posto:
                precisa_combustivel = True

        # Capacidade tem prioridade: volta ao depósito (recarrega carga + combustível)
        if precisa_carga:
            dist_volta = distancia_euclidiana(pos_atual, deposito)
            dist_total += dist_volta
            combustivel = autonomia
            cidades_visitadas = 0
            paradas += 1
            pos_atual = deposito
            dist_ate_cidade = distancia_euclidiana(deposito, cidade)
            # Após voltar ao depósito com tanque cheio, re-verificar combustível
            precisa_combustivel = False
            if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA:
                posto_prox = posto_mais_proximo(cidade, POSTOS_GASOLINA)
                dist_cidade_posto = distancia_euclidiana(cidade, posto_prox)
                if combustivel < dist_ate_cidade + dist_cidade_posto:
                    precisa_combustivel = True

        # Se ainda precisa de combustível (não foi ao depósito), vai ao posto mais próximo
        if precisa_combustivel:
            posto = posto_mais_proximo(pos_atual, POSTOS_GASOLINA)
            dist_ate_posto = distancia_euclidiana(pos_atual, posto)
            dist_total += dist_ate_posto
            combustivel = autonomia
            paradas += 1
            pos_atual = posto
            dist_ate_cidade = distancia_euclidiana(posto, cidade)

        if REABASTECIMENTO_ATIVO:
            combustivel -= dist_ate_cidade

        dist_total += dist_ate_cidade
        cidades_visitadas += 1
        pos_atual = cidade

    # Volta final à base — verificar se tem combustível suficiente
    dist_volta_final = distancia_euclidiana(pos_atual, deposito)
    if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA and combustivel < dist_volta_final:
        posto = posto_mais_proximo(pos_atual, POSTOS_GASOLINA)
        dist_ate_posto = distancia_euclidiana(pos_atual, posto)
        dist_total += dist_ate_posto
        combustivel = autonomia
        paradas += 1
        pos_atual = posto

    dist_total += distancia_euclidiana(pos_atual, deposito)
    return dist_total, paradas


def construir_waypoints_reabastecimento(rota, deposito, autonomia, capacidade=None):
    """Retorna a sequência real de pontos percorridos, incluindo paradas em
    postos de gasolina (reabastecimento) e/ou voltas ao depósito (carga).
    Retorna lista de (ponto, tipo) onde tipo é 'cidade', 'deposito' ou 'posto'."""
    if not rota:
        return [(deposito, 'deposito')]

    waypoints = [(deposito, 'deposito')]
    combustivel = autonomia
    cidades_visitadas = 0
    pos_atual = deposito

    for cidade in rota:
        dist_ate_cidade = distancia_euclidiana(pos_atual, cidade)
        precisa_carga = False
        precisa_combustivel = False

        if CAPACIDADE_CARGA and capacidade is not None:
            if cidades_visitadas >= capacidade:
                precisa_carga = True

        if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA:
            posto_prox = posto_mais_proximo(cidade, POSTOS_GASOLINA)
            dist_cidade_posto = distancia_euclidiana(cidade, posto_prox)
            if combustivel < dist_ate_cidade + dist_cidade_posto:
                precisa_combustivel = True

        # Capacidade: volta ao depósito (recarrega carga + combustível)
        if precisa_carga:
            waypoints.append((deposito, 'deposito'))
            combustivel = autonomia
            cidades_visitadas = 0
            pos_atual = deposito
            dist_ate_cidade = distancia_euclidiana(deposito, cidade)
            precisa_combustivel = False
            if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA:
                posto_prox = posto_mais_proximo(cidade, POSTOS_GASOLINA)
                dist_cidade_posto = distancia_euclidiana(cidade, posto_prox)
                if combustivel < dist_ate_cidade + dist_cidade_posto:
                    precisa_combustivel = True

        # Combustível: vai ao posto mais próximo
        if precisa_combustivel:
            posto = posto_mais_proximo(pos_atual, POSTOS_GASOLINA)
            waypoints.append((posto, 'posto'))
            combustivel = autonomia
            pos_atual = posto
            dist_ate_cidade = distancia_euclidiana(posto, cidade)

        if REABASTECIMENTO_ATIVO:
            combustivel -= dist_ate_cidade

        waypoints.append((cidade, 'cidade'))
        cidades_visitadas += 1
        pos_atual = cidade

    # Volta final — verificar combustível
    dist_volta_final = distancia_euclidiana(pos_atual, deposito)
    if REABASTECIMENTO_ATIVO and POSTOS_GASOLINA and combustivel < dist_volta_final:
        posto = posto_mais_proximo(pos_atual, POSTOS_GASOLINA)
        waypoints.append((posto, 'posto'))
        combustivel = autonomia
        pos_atual = posto

    waypoints.append((deposito, 'deposito'))
    return waypoints

def calcular_tempo_rota(rota, deposito, veiculo):
    """Calcula o tempo (horas) de uma rota considerando autonomia, capacidade e velocidade."""
    dist_efetiva, reab = calcular_distancia_efetiva(rota, deposito, veiculo['autonomia'], veiculo.get('capacidade'))
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

def calcular_penalidade_prioridade(rotas):
    """Calcula penalidade de prioridade: hospitais de alta prioridade que
    aparecem tarde na rota (posição alta) recebem penalidade proporcional.
    Quanto mais cedo na rota um hospital urgente for visitado, menor a penalidade."""
    if not PRIORIDADE_ATIVA:
        return 0.0
    penalidade = 0.0
    for rota in rotas:
        n = len(rota)
        if n == 0:
            continue
        for pos, cidade in enumerate(rota):
            info = mapa_prioridade.get(cidade)
            if info is None:
                continue
            peso_nivel = PESOS_NIVEL_PRIORIDADE.get(info['prioridade'], 0.0)
            # Fração normalizada da posição (0.0 = primeiro, 1.0 = último)
            fracao_pos = pos / max(n - 1, 1)
            # Penalidade = peso do nível × posição relativa na rota
            # Hospitais urgentes no início → penalidade baixa
            # Hospitais urgentes no final → penalidade alta
            penalidade += peso_nivel * fracao_pos
    return penalidade


def calcular_fitness_mtsp(cromossomo, deposito, n_veiculos, peso_balanceamento=PESO_BALANCEAMENTO):
    """Calcula o fitness do mTSP conforme o OBJETIVO configurado.

    OBJETIVO='distancia' → fitness = max(dist_ef) + peso * total_dist_ef
    OBJETIVO='tempo'     → fitness = max(tempo) + peso * total_tempo
    OBJETIVO='hibrido'   → combina as duas métricas normalizadas por VELOCIDADE_REF:
                           dist_component  (km)
                           tempo_component (horas * VELOCIDADE_REF → km-equivalente)
                           fitness = (1 - ALFA) * dist_component + ALFA * tempo_component

    Quando PRIORIDADE_ATIVA=True, soma penalidade de prioridade ao fitness.
    """
    rotas = dividir_rota(cromossomo, n_veiculos)
    dist_efetivas = []
    tempos = []
    for idx, rota in enumerate(rotas):
        veiculo = VEICULOS[idx]
        dist_ef, _ = calcular_distancia_efetiva(rota, deposito, veiculo['autonomia'], veiculo.get('capacidade'))
        tempo, _, _ = calcular_tempo_rota(rota, deposito, veiculo)
        dist_efetivas.append(dist_ef)
        tempos.append(tempo)

    # Penalidade de prioridade (0 se desativada)
    pen_prioridade = calcular_penalidade_prioridade(rotas) * PESO_PRIORIDADE

    if OBJETIVO == 'tempo':
        amplitude = max(tempos) - min(tempos)
        # O objetivo é minimizar o tempo máximo + forte penalidade no desbalanceamento
        return max(tempos) + (peso_balanceamento * sum(tempos)) + (amplitude * 5.0) + pen_prioridade

    elif OBJETIVO == 'hibrido':
        # Componente de distância (km)
        dist_component = max(dist_efetivas) + peso_balanceamento * sum(dist_efetivas)
        # Componente de tempo convertido para km-equivalente (h × km/h)
        amplitude_tempo = max(tempos) - min(tempos)
        tempo_component = (max(tempos) + peso_balanceamento * sum(tempos) + amplitude_tempo * 5.0) * VELOCIDADE_REF
        return (1 - ALFA_HIBRIDO) * dist_component + ALFA_HIBRIDO * tempo_component + pen_prioridade

    else:  # 'distancia'
        return max(dist_efetivas) + peso_balanceamento * sum(dist_efetivas) + pen_prioridade


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
def dois_opt(rota, deposito, autonomia, capacidade=None):
    """Aplica 2-opt numa sub-rota usando distância EFETIVA (com reabastecimento e capacidade)."""
    if len(rota) < 3:
        return rota
    melhorou = True
    melhor_rota = list(rota)
    melhor_dist, _ = calcular_distancia_efetiva(melhor_rota, deposito, autonomia, capacidade)
    while melhorou:
        melhorou = False
        for i in range(len(melhor_rota) - 1):
            for j in range(i + 2, len(melhor_rota)):
                nova_rota = melhor_rota[:i] + melhor_rota[i:j+1][::-1] + melhor_rota[j+1:]
                nova_dist, _ = calcular_distancia_efetiva(nova_rota, deposito, autonomia, capacidade)
                if nova_dist < melhor_dist:
                    melhor_rota = nova_rota
                    melhor_dist = nova_dist
                    melhorou = True
    return melhor_rota


def aplicar_2opt_mtsp(cromossomo, deposito, n_veiculos):
    """Aplica 2-opt em cada sub-rota do cromossomo usando a autonomia e capacidade do veículo."""
    rotas = dividir_rota(cromossomo, n_veiculos)
    rotas_otimizadas = []
    for idx, rota in enumerate(rotas):
        veiculo = VEICULOS[idx]
        rotas_otimizadas.append(dois_opt(rota, deposito, veiculo['autonomia'], veiculo.get('capacidade')))
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
    # Desenhar cidades com cor por prioridade (se ativa)
    if PRIORIDADE_ATIVA:
        _cores_prio = {0: VERMELHO, 1: AMARELO, 2: (0, 180, 0)}  # Alta=vermelho, Média=amarelo, Baixa=verde
        for _cidade in cidades_sem_deposito:
            _info = mapa_prioridade.get(_cidade)
            _cor_c = _cores_prio.get(_info['prioridade'], VERMELHO) if _info else VERMELHO
            pygame.draw.circle(tela, _cor_c, _cidade, RAIO_NO)
    else:
        draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
    # Depósito com cor especial e tamanho maior
    pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
    pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)

    # Desenhar postos de gasolina (azul) no mapa
    if REABASTECIMENTO_ATIVO:
        for _posto in POSTOS_GASOLINA:
            pygame.draw.circle(tela, AZUL, _posto, RAIO_NO + 3)
            pygame.draw.circle(tela, PRETO, _posto, RAIO_NO + 3, 2)

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

        # Construir waypoints reais (com paradas de reabastecimento/carga)
        waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, veiculo['autonomia'], veiculo.get('capacidade'))
        pontos = [wp[0] for wp in waypoints]
        tipos = [wp[1] for wp in waypoints]

        # Desenhar segmento a segmento
        for i in range(len(pontos) - 1):
            p1, p2 = pontos[i], pontos[i + 1]
            # Segmento de reabastecimento/depósito/posto: tracejado e mais fino
            if tipos[i] in ('deposito', 'posto') or tipos[i + 1] in ('deposito', 'posto'):
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
GERACOES_REFINAMENTO = 300
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
    if PRIORIDADE_ATIVA:
        _cores_prio = {0: VERMELHO, 1: AMARELO, 2: (0, 180, 0)}
        for _cidade in cidades_sem_deposito:
            _info = mapa_prioridade.get(_cidade)
            _cor_c = _cores_prio.get(_info['prioridade'], VERMELHO) if _info else VERMELHO
            pygame.draw.circle(tela, _cor_c, _cidade, RAIO_NO)
    else:
        draw_cities(tela, cidades_sem_deposito, VERMELHO, RAIO_NO)
    pygame.draw.circle(tela, AMARELO, DEPOSITO, RAIO_NO + 5)
    pygame.draw.circle(tela, PRETO, DEPOSITO, RAIO_NO + 5, 3)
    if REABASTECIMENTO_ATIVO:
        for _posto in POSTOS_GASOLINA:
            pygame.draw.circle(tela, AZUL, _posto, RAIO_NO + 3)
            pygame.draw.circle(tela, PRETO, _posto, RAIO_NO + 3, 2)
    rotas_temp = dividir_rota(melhor_solucao_final, N_VEICULOS)
    for idx_rota, rota_t in enumerate(rotas_temp):
        cor_veiculo = CORES_VEICULOS[idx_rota % len(CORES_VEICULOS)]
        autonomia_v = VEICULOS[idx_rota]['autonomia']
        capacidade_v = VEICULOS[idx_rota].get('capacidade')
        waypoints = construir_waypoints_reabastecimento(rota_t, DEPOSITO, autonomia_v, capacidade_v)
        for p_idx in range(len(waypoints) - 1):
            inicio_p, t_inicio = waypoints[p_idx]
            fim_p, t_fim = waypoints[p_idx + 1]
            if t_inicio in ('deposito', 'posto') or t_fim in ('deposito', 'posto'):
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
if REABASTECIMENTO_ATIVO:
    for _posto in POSTOS_GASOLINA:
        pygame.draw.circle(tela, AZUL, _posto, RAIO_NO + 3)
        pygame.draw.circle(tela, PRETO, _posto, RAIO_NO + 3, 2)

for idx, rota in enumerate(rotas_finais):
    if len(rota) == 0:
        continue
    cor = CORES_VEICULOS[idx % len(CORES_VEICULOS)]
    autonomia_v = VEICULOS[idx]['autonomia']
    capacidade_v = VEICULOS[idx].get('capacidade')
    waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, autonomia_v, capacidade_v)

    for p_idx in range(len(waypoints) - 1):
        inicio_p, t_inicio = waypoints[p_idx]
        fim_p, t_fim = waypoints[p_idx + 1]
        if t_inicio in ('deposito', 'posto') or t_fim in ('deposito', 'posto'):
            # Trecho de reabastecimento/carga: linha fina + marcador no ponto médio
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
    if PRIORIDADE_ATIVA:
        _labels_p = {0: '🔴Alta', 1: '🟡Média', 2: '🟢Baixa'}
        for cidade in rota:
            info = mapa_prioridade.get(cidade)
            if info:
                idx_c = localizacoes_cidades.index(cidade)
                print(f"    → {info['nome']} [{_labels_p[info['prioridade']]}] (cidade {idx_c+1})")
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
    # === Agente OpenAI - Roteiro Individual por Motorista ===
    print("\n=== Gerando roteiros individuais por motorista via ChatGPT... ===\n")

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada no arquivo .env")
    else:
        client = OpenAI(api_key=api_key)

        _labels_prio = {0: '🔴 ALTA', 1: '🟡 MÉDIA', 2: '🟢 BAIXA'}

        # --- Montar roteiro detalhado passo a passo por veículo ---
        # Usa os waypoints reais (inclui postos de reabastecimento e retornos ao depósito)
        # para gerar um guia sequencial completo para cada motorista.
        texto_roteiros = ""
        for idx, rota in enumerate(rotas_finais):
            v = VEICULOS[idx]
            autonomia_v = v['autonomia']
            capacidade_v = v.get('capacidade')
            velocidade_v = v['velocidade']

            waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, autonomia_v, capacidade_v)

            cap_str = f" | Capacidade: {capacidade_v} entregas/viagem" if CAPACIDADE_CARGA and capacidade_v else ""
            texto_roteiros += f"\n{'=' * 60}\n"
            texto_roteiros += f"VEÍCULO {idx + 1} — {v['tipo'].upper()}\n"
            texto_roteiros += f"Autonomia: {autonomia_v}km | Velocidade: {velocidade_v}km/h{cap_str}\n"
            texto_roteiros += f"Total: {len(rota)} entregas | {round(dist_ef_final[idx], 2)}km | ~{round(tempos_final[idx], 2)}h\n"
            texto_roteiros += f"{'=' * 60}\n"
            texto_roteiros += "ROTEIRO DE PARADAS:\n\n"

            dist_acum = 0.0
            tempo_acum = 0.0
            posto_num = 0
            viagem_num = 1
            entrega_num = 0
            pos_prev = None

            for wp_idx, (ponto, tipo) in enumerate(waypoints):
                if wp_idx == 0:
                    texto_roteiros += "  🏁  PARTIDA — Depósito Central\n"
                    pos_prev = ponto
                    continue

                dist_seg = distancia_euclidiana(pos_prev, ponto)
                dist_acum += dist_seg
                tempo_acum += dist_seg / velocidade_v
                pos_prev = ponto

                if tipo == 'deposito':
                    if wp_idx == len(waypoints) - 1:
                        texto_roteiros += (
                            f"  🏠  RETORNO FINAL ao Depósito Central"
                            f" | +{round(dist_seg, 1)}km"
                            f" | Total acumulado: {round(dist_acum, 1)}km"
                            f" | Tempo estimado: ~{round(tempo_acum, 2)}h\n"
                        )
                    else:
                        viagem_num += 1
                        texto_roteiros += (
                            f"  🔄  RETORNO AO DEPÓSITO para recarga de carga"
                            f" (início da viagem {viagem_num})"
                            f" | +{round(dist_seg, 1)}km"
                            f" | Total acumulado: {round(dist_acum, 1)}km"
                            f" | Tempo estimado: ~{round(tempo_acum, 2)}h\n"
                        )
                elif tipo == 'posto':
                    posto_num += 1
                    idx_posto = (POSTOS_GASOLINA.index(ponto) + 1) if ponto in POSTOS_GASOLINA else posto_num
                    texto_roteiros += (
                        f"  ⛽  PARADA OBRIGATÓRIA — Abastecimento no Posto #{idx_posto}"
                        f" | +{round(dist_seg, 1)}km"
                        f" | Total acumulado: {round(dist_acum, 1)}km"
                        f" | Tempo estimado: ~{round(tempo_acum, 2)}h\n"
                    )
                else:  # 'cidade'
                    entrega_num += 1
                    idx_cidade = localizacoes_cidades.index(ponto)
                    nome_hosp = att_48_hospitals[idx_cidade]
                    prio = _labels_prio.get(att_48_priorities[idx_cidade], '?')
                    alerta = "  ⚠️  ATENDA PRIMEIRO!" if att_48_priorities[idx_cidade] == 0 else ""
                    texto_roteiros += (
                        f"  📦  Entrega {entrega_num:02d} — {nome_hosp} [{prio}]{alerta}"
                        f" | +{round(dist_seg, 1)}km"
                        f" | Total acumulado: {round(dist_acum, 1)}km"
                        f" | Tempo estimado: ~{round(tempo_acum, 2)}h\n"
                    )

            texto_roteiros += "\n"

        # --- Contexto operacional para o prompt ---
        contexto_op = []
        if PRIORIDADE_ATIVA:
            contexto_op.append(
                "Sistema de PRIORIDADES ATIVO: hospitais 🔴 ALTA devem ser atendidos o mais cedo possível na rota."
            )
        if REABASTECIMENTO_ATIVO:
            contexto_op.append(
                "Sistema de REABASTECIMENTO ATIVO: o motorista DEVE parar no posto indicado para abastecer "
                "antes de prosseguir — ignorar essa parada pode deixar o veículo sem combustível."
            )
        if CAPACIDADE_CARGA:
            contexto_op.append(
                "Sistema de CAPACIDADE DE CARGA ATIVO: ao atingir o limite de entregas por viagem, "
                "o motorista DEVE retornar ao depósito para buscar mais itens antes de continuar."
            )
        contexto_str = "\n".join(f"- {c}" for c in contexto_op) if contexto_op else "- Operação padrão sem restrições adicionais."

        prompt = f"""Você é um assistente de logística responsável por preparar os roteiros de entrega para os motoristas de uma frota hospitalar.

Com base nos dados estruturados abaixo, escreva um roteiro claro, numerado e fácil de ler para cada motorista executar sua rota.
Use linguagem direta e imperativa, como um GPS textual ("Siga para...", "Pare em...", "Retorne ao depósito...").
O motorista não conhece o sistema — escreva como se fosse a única instrução que ele vai receber.

## Contexto da Operação
- Data da rota: {timestamp[:10]}
- Frota em operação: {N_VEICULOS} veículo(s) — {N_CARROS} carro(s) e {N_MOTOS} moto(s)
- Ponto de partida e chegada: Depósito Central
{contexto_str}

## Dados dos Roteiros
{texto_roteiros}

## Como formatar o roteiro de cada motorista
Para cada veículo, produza um bloco independente contendo:

1. **Cabeçalho**: número e tipo do veículo, autonomia, velocidade média e resumo da missão
   (quantas entregas, distância total estimada, tempo total estimado)

2. **Roteiro numerado passo a passo**: para cada parada, informe:
   - Número da parada
   - Tipo: Entrega 📦, Abastecimento ⛽, Retorno para recarga 🔄 ou Retorno final 🏠
   - Nome exato do hospital ou descrição do ponto de parada
   - Prioridade da entrega (se aplicável) — destaque com alerta os hospitais de prioridade 🔴 ALTA
   - Distância parcial desde a parada anterior e distância total acumulada
   - Tempo estimado acumulado desde a partida

3. **Instruções especiais em destaque**:
   - Para paradas de abastecimento ⛽: avise que o veículo DEVE abastecer — não é opcional
   - Para retornos ao depósito 🔄: avise que o motorista deve buscar mais itens antes de continuar

4. **Resumo final**: total de entregas realizadas, distância total percorrida e tempo total estimado

Responda apenas com os roteiros, sem comentários adicionais ou análises, em português brasileiro."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Você é um assistente de logística que prepara roteiros práticos e objetivos "
                        "para motoristas de entrega em hospitais. Seja direto, claro e use linguagem imperativa."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3500
            )
            resposta = response.choices[0].message.content

            print("=" * 60)
            print("ROTEIROS DOS MOTORISTAS — GERADO POR ChatGPT")
            print("=" * 60)
            print(resposta)
            print("=" * 60)

            # Salvar roteiros em arquivo de texto
            nome_roteiro = f"rotas/roteiro_motoristas_{timestamp}.txt"
            with open(nome_roteiro, 'w', encoding='utf-8') as f:
                f.write("ROTEIROS DOS MOTORISTAS\n")
                f.write(f"Gerado em: {timestamp}\n")
                f.write("=" * 60 + "\n\n")
                f.write(resposta)
            print(f"Roteiros salvos em: {nome_roteiro}")

        except Exception as e:
            print(f"Erro ao chamar a API OpenAI: {e}")

        # --- Gerar PDF com roteiro imprimível por motorista ---
        # Gerado a partir dos dados estruturados (independente do resultado do GPT)
        try:
            import html
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle,
                Paragraph, Spacer, PageBreak, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_RIGHT

            PAGE_W = A4[0] - 3 * cm   # usable width (1.5cm margin each side)
            _COL_W = [0.7*cm, 2.3*cm, 7.5*cm, 1.9*cm, 1.7*cm, 2.0*cm, 1.9*cm]  # total ≈ 18cm

            # Cores por tipo de linha
            COR_ALTA      = colors.HexColor('#FFEBEE')
            COR_MEDIA     = colors.HexColor('#FFF8E1')
            COR_BAIXA     = colors.HexColor('#F1F8E9')
            COR_POSTO     = colors.HexColor('#FFF9C4')
            COR_DEPOSITO  = colors.HexColor('#E3F2FD')
            COR_HEADER    = colors.HexColor('#1A237E')
            COR_CARRO     = colors.HexColor('#1565C0')
            COR_MOTO      = colors.HexColor('#E65100')

            def _s(text, size=8, bold=False, color=None, align='LEFT'):
                """Cria um Paragraph simples com estilo embutido."""
                font = 'Helvetica-Bold' if bold else 'Helvetica'
                col = color if color else '#000000'
                alignment = TA_CENTER if align == 'CENTER' else (TA_RIGHT if align == 'RIGHT' else 0)
                return Paragraph(
                    html.escape(str(text)),
                    ParagraphStyle('_', fontName=font, fontSize=size,
                                   textColor=colors.HexColor(col),
                                   alignment=alignment, leading=size + 2)
                )

            nome_pdf = f"rotas/roteiro_motoristas_{timestamp}.pdf"
            doc = SimpleDocTemplate(
                nome_pdf, pagesize=A4,
                rightMargin=1.5*cm, leftMargin=1.5*cm,
                topMargin=1.5*cm, bottomMargin=1.5*cm
            )

            elementos_pdf = []

            # ── Capa do lote ──────────────────────────────────────────────
            elementos_pdf.append(
                Paragraph('ROTEIROS DE ENTREGA', ParagraphStyle(
                    'capa', fontName='Helvetica-Bold', fontSize=22,
                    textColor=COR_HEADER, spaceAfter=4))
            )
            elementos_pdf.append(
                Paragraph(
                    f'Data: {timestamp[:10]}   |   '
                    f'Frota: {N_VEICULOS} veiculo(s)   |   '
                    f'{len(rotas_finais[0]) + sum(len(r) for r in rotas_finais[1:])} hospitais no total',
                    ParagraphStyle('sub', fontName='Helvetica', fontSize=9,
                                   textColor=colors.HexColor('#555555'), spaceAfter=6))
            )
            elementos_pdf.append(HRFlowable(width='100%', thickness=2, color=COR_HEADER, spaceAfter=10))

            # ── Um bloco por veículo ──────────────────────────────────────
            _labels_prio_pdf = {0: 'ALTA', 1: 'MEDIA', 2: 'BAIXA'}

            for idx, rota in enumerate(rotas_finais):
                v = VEICULOS[idx]
                autonomia_v  = v['autonomia']
                capacidade_v = v.get('capacidade')
                velocidade_v = v['velocidade']
                cor_v = COR_CARRO if 'Carro' in v['tipo'] else COR_MOTO

                waypoints = construir_waypoints_reabastecimento(rota, DEPOSITO, autonomia_v, capacidade_v)

                # Cabeçalho do veículo
                cap_str = f"  |  Cap.: {capacidade_v} entr./viagem" if CAPACIDADE_CARGA and capacidade_v else ''
                cab_dados = [[
                    Paragraph(
                        f"<b>VEICULO {idx+1} — {html.escape(v['tipo'].upper())}</b>",
                        ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=14,
                                       textColor=colors.white, leading=16)
                    ),
                    Paragraph(
                        f"<b>{len(rota)} entregas  |  {round(dist_ef_final[idx],1)} km  |  ~{round(tempos_final[idx],2)} h</b>",
                        ParagraphStyle('ch2', fontName='Helvetica-Bold', fontSize=10,
                                       textColor=colors.white, alignment=TA_RIGHT, leading=12)
                    ),
                ]]
                tabela_cab = Table(cab_dados, colWidths=[PAGE_W * 0.6, PAGE_W * 0.4])
                tabela_cab.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), cor_v),
                    ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING',   (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
                ]))
                elementos_pdf.append(tabela_cab)
                elementos_pdf.append(
                    Paragraph(
                        f'Autonomia: {autonomia_v} km  |  Velocidade media: {velocidade_v} km/h'
                        f'{html.escape(cap_str)}  |  Paradas extras: {reab_final[idx]}',
                        ParagraphStyle('info', fontName='Helvetica', fontSize=8,
                                       textColor=colors.HexColor('#444444'),
                                       spaceBefore=3, spaceAfter=5)
                    )
                )

                # Cabeçalho da tabela de paradas
                header_row = [
                    _s('#',       bold=True, color='#FFFFFF', align='CENTER'),
                    _s('TIPO',    bold=True, color='#FFFFFF', align='CENTER'),
                    _s('DESTINO / ACAO',  bold=True, color='#FFFFFF'),
                    _s('PRIORIDADE',      bold=True, color='#FFFFFF', align='CENTER'),
                    _s('+KM',     bold=True, color='#FFFFFF', align='RIGHT'),
                    _s('KM TOTAL',bold=True, color='#FFFFFF', align='RIGHT'),
                    _s('TEMPO',   bold=True, color='#FFFFFF', align='RIGHT'),
                ]
                dados_tabela = [header_row]
                estilos_linhas = [
                    ('BACKGROUND', (0, 0), (-1, 0), COR_HEADER),
                    ('LINEBELOW',  (0, 0), (-1, 0), 1, COR_HEADER),
                ]

                dist_acum = 0.0
                tempo_acum = 0.0
                viagem_num = 1
                entrega_num = 0
                posto_num = 0
                pos_prev = None

                for wp_idx, (ponto, tipo) in enumerate(waypoints):
                    row_i = len(dados_tabela)  # real row index in table (for TableStyle)

                    if wp_idx == 0:
                        pos_prev = ponto
                        dados_tabela.append([
                            _s('—', align='CENTER'),
                            _s('PARTIDA', bold=True),
                            _s('Deposito Central'),
                            _s('—', align='CENTER'),
                            _s('—', align='RIGHT'),
                            _s('0,0 km', align='RIGHT'),
                            _s('0,00 h', align='RIGHT'),
                        ])
                        estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_DEPOSITO))
                        continue

                    dist_seg   = distancia_euclidiana(pos_prev, ponto)
                    dist_acum += dist_seg
                    tempo_acum += dist_seg / velocidade_v
                    pos_prev   = ponto

                    if tipo == 'deposito':
                        if wp_idx == len(waypoints) - 1:
                            dados_tabela.append([
                                _s('—', align='CENTER'),
                                _s('CHEGADA', bold=True),
                                _s('Deposito Central — FIM DA ROTA'),
                                _s('—', align='CENTER'),
                                _s(f'{round(dist_seg,1)} km', align='RIGHT'),
                                _s(f'{round(dist_acum,1)} km', align='RIGHT'),
                                _s(f'{round(tempo_acum,2)} h', align='RIGHT'),
                            ])
                            estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_DEPOSITO))
                        else:
                            viagem_num += 1
                            dados_tabela.append([
                                _s('—', align='CENTER'),
                                _s('RECARREGAR', bold=True, color='#BF360C'),
                                _s(f'Retorne ao Deposito — buscar itens (viagem {viagem_num})'),
                                _s('—', align='CENTER'),
                                _s(f'{round(dist_seg,1)} km', align='RIGHT'),
                                _s(f'{round(dist_acum,1)} km', align='RIGHT'),
                                _s(f'{round(tempo_acum,2)} h', align='RIGHT'),
                            ])
                            estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), colors.HexColor('#FFE0B2')))
                            estilos_linhas.append(('FONTNAME', (0, row_i), (-1, row_i), 'Helvetica-Bold'))

                    elif tipo == 'posto':
                        posto_num += 1
                        idx_posto = (POSTOS_GASOLINA.index(ponto) + 1) if ponto in POSTOS_GASOLINA else posto_num
                        dados_tabela.append([
                            _s('—', align='CENTER'),
                            _s('ABASTECER', bold=True, color='#F57F17'),
                            _s(f'Posto de Gasolina #{idx_posto} — OBRIGATORIO abastecer'),
                            _s('—', align='CENTER'),
                            _s(f'{round(dist_seg,1)} km', align='RIGHT'),
                            _s(f'{round(dist_acum,1)} km', align='RIGHT'),
                            _s(f'{round(tempo_acum,2)} h', align='RIGHT'),
                        ])
                        estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_POSTO))
                        estilos_linhas.append(('FONTNAME', (0, row_i), (-1, row_i), 'Helvetica-Bold'))

                    else:  # cidade
                        entrega_num += 1
                        idx_cidade = localizacoes_cidades.index(ponto)
                        nome_hosp  = att_48_hospitals[idx_cidade]
                        prio_num   = att_48_priorities[idx_cidade] if PRIORIDADE_ATIVA else 2
                        prio_str   = _labels_prio_pdf.get(prio_num, '?')

                        alerta_txt = f'{nome_hosp}  *** ATENDER IMEDIATAMENTE ***' if prio_num == 0 else nome_hosp
                        dados_tabela.append([
                            _s(str(entrega_num), bold=(prio_num == 0), align='CENTER'),
                            _s('ENTREGA', bold=(prio_num == 0)),
                            _s(alerta_txt, bold=(prio_num == 0)),
                            _s(prio_str, bold=(prio_num == 0), align='CENTER'),
                            _s(f'{round(dist_seg,1)} km', align='RIGHT'),
                            _s(f'{round(dist_acum,1)} km', align='RIGHT'),
                            _s(f'{round(tempo_acum,2)} h', align='RIGHT'),
                        ])
                        if prio_num == 0:
                            estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_ALTA))
                            estilos_linhas.append(('FONTNAME',   (0, row_i), (-1, row_i), 'Helvetica-Bold'))
                        elif prio_num == 1:
                            estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_MEDIA))
                        else:
                            estilos_linhas.append(('BACKGROUND', (0, row_i), (-1, row_i), COR_BAIXA))

                # Montar tabela de paradas
                tabela_paradas = Table(dados_tabela, colWidths=_COL_W, repeatRows=1)
                estilo_tabela = TableStyle([
                    ('FONTSIZE',      (0, 0), (-1, -1), 8),
                    ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING',    (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
                    ('GRID',          (0, 0), (-1, -1), 0.25, colors.HexColor('#CCCCCC')),
                    ('LINEBELOW',     (0, 0), (-1, 0), 1.2, COR_HEADER),
                    ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFAFA')]),
                ])
                for est in estilos_linhas:
                    estilo_tabela.add(*est)
                tabela_paradas.setStyle(estilo_tabela)
                elementos_pdf.append(tabela_paradas)

                # Legenda de cores
                elementos_pdf.append(Spacer(1, 0.25*cm))
                legenda_dados = [[
                    Paragraph('<b>Legenda:</b>', ParagraphStyle('lg', fontName='Helvetica-Bold', fontSize=7)),
                    Paragraph('Prioridade ALTA', ParagraphStyle('lg', fontName='Helvetica', fontSize=7, backColor=COR_ALTA)),
                    Paragraph('Prioridade MEDIA', ParagraphStyle('lg', fontName='Helvetica', fontSize=7, backColor=COR_MEDIA)),
                    Paragraph('Prioridade BAIXA', ParagraphStyle('lg', fontName='Helvetica', fontSize=7, backColor=COR_BAIXA)),
                    Paragraph('Abastecer', ParagraphStyle('lg', fontName='Helvetica', fontSize=7, backColor=COR_POSTO)),
                    Paragraph('Deposito', ParagraphStyle('lg', fontName='Helvetica', fontSize=7, backColor=COR_DEPOSITO)),
                ]]
                tabela_leg = Table(legenda_dados, colWidths=[2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.0*cm, 1.8*cm])
                tabela_leg.setStyle(TableStyle([
                    ('FONTSIZE',      (0, 0), (-1, -1), 7),
                    ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING',    (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 4),
                    ('BACKGROUND',    (1, 0), (1, 0), COR_ALTA),
                    ('BACKGROUND',    (2, 0), (2, 0), COR_MEDIA),
                    ('BACKGROUND',    (3, 0), (3, 0), COR_BAIXA),
                    ('BACKGROUND',    (4, 0), (4, 0), COR_POSTO),
                    ('BACKGROUND',    (5, 0), (5, 0), COR_DEPOSITO),
                    ('BOX',           (1, 0), (-1, 0), 0.3, colors.HexColor('#AAAAAA')),
                ]))
                elementos_pdf.append(tabela_leg)

                # Resumo do veículo
                elementos_pdf.append(Spacer(1, 0.2*cm))
                resumo_dados = [[
                    _s(f'RESUMO: {len(rota)} entregas realizadas  |  '
                       f'Distancia total: {round(dist_ef_final[idx], 1)} km  |  '
                       f'Tempo estimado: ~{round(tempos_final[idx], 2)} h  |  '
                       f'Paradas extras: {reab_final[idx]}',
                       bold=True, size=8)
                ]]
                tabela_resumo = Table(resumo_dados, colWidths=[PAGE_W])
                tabela_resumo.setStyle(TableStyle([
                    ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#EEEEEE')),
                    ('TOPPADDING',    (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                    ('BOX',           (0, 0), (-1, -1), 0.5, COR_HEADER),
                ]))
                elementos_pdf.append(tabela_resumo)

                if idx < len(rotas_finais) - 1:
                    elementos_pdf.append(PageBreak())

            doc.build(elementos_pdf)
            print(f"PDF dos roteiros salvo: {nome_pdf}")

        except ImportError:
            print("AVISO: reportlab nao instalado. Execute: pip install reportlab")
        except Exception as e_pdf:
            print(f"Erro ao gerar PDF: {e_pdf}")

else:
    print("Relatório GPT desativado.")

# Encerrar programa
pygame.quit()
sys.exit()
