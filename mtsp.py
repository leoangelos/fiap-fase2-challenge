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
PROBABILIDADE_MUTACAO = 0.6
GERACOES_SEM_MELHORA_PARA_PARAR = 800  # Encerra se não melhorar após N gerações
N_VEICULOS = 3  # Número de veículos
HEURISTICA = 2  # 1 = Vizinho Mais Próximo | 2 = Convex Hull
PESO_BALANCEAMENTO = 0.2  # Penalidade por desbalanceamento entre rotas

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


def calcular_distancia_rota(rota, deposito):
    """Calcula a distância de uma sub-rota: depósito → cidades → depósito."""
    if len(rota) == 0:
        return 0
    dist = distancia_euclidiana(deposito, rota[0])
    for i in range(len(rota) - 1):
        dist += distancia_euclidiana(rota[i], rota[i + 1])
    dist += distancia_euclidiana(rota[-1], deposito)
    return dist


def calcular_fitness_mtsp(cromossomo, deposito, n_veiculos, peso_balanceamento=PESO_BALANCEAMENTO):
    """Calcula o fitness do mTSP usando abordagem minimax.
    Minimiza a rota mais longa + fração da distância total.
    fitness = max(distancias) + peso * distancia_total"""
    rotas = dividir_rota(cromossomo, n_veiculos)
    distancias = [calcular_distancia_rota(rota, deposito) for rota in rotas]

    distancia_total = sum(distancias)
    max_distancia = max(distancias) if distancias else 0

    # Minimax: prioriza reduzir a rota mais longa
    # peso_balanceamento controla quanto a distância total importa
    #   peso=0.0 → só minimiza a rota mais longa
    #   peso=1.0 → igual peso entre max e total
    return max_distancia + peso_balanceamento * distancia_total


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


# --- Operadores de Mutação Avançados para mTSP ---
def mutacao_inversao(cromossomo):
    """Inverte um segmento aleatório do cromossomo (movimento 2-opt)."""
    n = len(cromossomo)
    if n < 3:
        return cromossomo
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, min(i + n // 3, n - 1))  # Limita tamanho do segmento
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
def dois_opt(rota, deposito):
    """Aplica 2-opt numa sub-rota para reduzir distância."""
    if len(rota) < 3:
        return rota
    melhorou = True
    melhor_rota = list(rota)
    while melhorou:
        melhorou = False
        for i in range(len(melhor_rota) - 1):
            for j in range(i + 2, len(melhor_rota)):
                nova_rota = melhor_rota[:i] + melhor_rota[i:j+1][::-1] + melhor_rota[j+1:]
                if calcular_distancia_rota(nova_rota, deposito) < calcular_distancia_rota(melhor_rota, deposito):
                    melhor_rota = nova_rota
                    melhorou = True
    return melhor_rota


def aplicar_2opt_mtsp(cromossomo, deposito, n_veiculos):
    """Aplica 2-opt em cada sub-rota do cromossomo."""
    rotas = dividir_rota(cromossomo, n_veiculos)
    rotas_otimizadas = [dois_opt(rota, deposito) for rota in rotas]
    novo_cromossomo = []
    for rota in rotas_otimizadas:
        novo_cromossomo.extend(rota)
    return novo_cromossomo
# --- Fim 2-opt ---


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

    # Elitismo: manter os top 3
    nova_populacao = populacao[:3]

    # Aplicar 2-opt periodicamente no melhor indivíduo
    if geracao % 50 == 0:
        cromossomo_otimizado = aplicar_2opt_mtsp(populacao[0], DEPOSITO, N_VEICULOS)
        nova_populacao[0] = cromossomo_otimizado

    while len(nova_populacao) < TAMANHO_POPULACAO:

        # seleção baseada na probabilidade do fitness
        probabilidade = 1 / np.array(fitness_populacao)
        pai1, pai2 = random.choices(populacao, weights=probabilidade, k=2)

        # crossover (corrigido: pai1 × pai2 em vez de pai × ele mesmo)
        filho1 = order_crossover(pai1, pai2)
        filho2 = order_crossover(pai2, pai1)

        # mutação avançada para mTSP
        filho1 = mutacao_mtsp(filho1, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)
        filho2 = mutacao_mtsp(filho2, PROBABILIDADE_MUTACAO, N_VEICULOS, DEPOSITO)

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
distancias_finais = []
for idx, rota in enumerate(rotas_finais):
    dist = calcular_distancia_rota(rota, DEPOSITO)
    distancias_finais.append(dist)
    print(f"Veículo {idx + 1}: {len(rota)} cidades, distância = {round(dist, 2)}")
distancia_total_real = sum(distancias_finais)
media_dist = distancia_total_real / len(distancias_finais)
desvio_final = (sum((d - media_dist)**2 for d in distancias_finais) / len(distancias_finais)) ** 0.5
print(f"Distância total: {round(distancia_total_real, 2)}")
print(f"Desvio padrão entre rotas: {round(desvio_final, 2)}")
print(f"Balanço: {round(min(distancias_finais), 2)} - {round(max(distancias_finais), 2)} (diferença: {round(max(distancias_finais) - min(distancias_finais), 2)})")

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
        dist = calcular_distancia_rota(rota, DEPOSITO)
        descricao_rotas += f"\n### Veículo {idx + 1} ({len(rota)} cidades, distância: {round(dist, 2)})\n"
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
- Número de veículos: {N_VEICULOS}
- Depósito (ponto de partida/chegada): Depósito do Hospital 1 ({att_48_cities_locations[0][0]}, {att_48_cities_locations[0][1]})
- Algoritmo: Genético com heurística {'Vizinho Mais Próximo' if HEURISTICA == 1 else 'Convex Hull'}
- Gerações: {geracao}
- Fitness (minimax): {round(melhor_fitness_final, 2)}

## Resultados das Rotas
{descricao_rotas}

## Estatísticas
- Distância total: {round(distancia_total_real, 2)}
- Desvio padrão entre rotas: {round(desvio_final, 2)}
- Diferença entre maior e menor rota: {round(max(distancias_finais) - min(distancias_finais), 2)}

## Instruções
1. Descreva o trajeto de cada veículo de forma clara e objetiva
2. Analise o balanceamento das rotas (distâncias dos veículos estão equilibradas?)
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

# encerrar programa
pygame.quit()
sys.exit()
