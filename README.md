# Solucionador TSP e mTSP com Algoritmo Genético

Este repositório contém uma implementação em Python para resolver o **Problema do Caixeiro Viajante (TSP)** e sua variante com **múltiplos veículos heterogêneos (mTSP)**, utilizando **Algoritmo Genético (AG)**. O objetivo é distribuir a visitação de hospitais entre uma frota mista de carros e motos, minimizando distância total ou equilibrando o tempo de chegada.

## Visão Geral

O AG evolui iterativamente uma população de soluções (permutações de cidades) usando seleção, cruzamento (OX1) e mutação. O elitismo garante que a **melhor solução global** nunca seja perdida entre gerações. Ao convergir (N gerações sem melhora), os resultados são enviados automaticamente ao **agente ChatGPT** para análise e geração do roteiro de entrega.

---

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `tsp.py` | Solucionador TSP (1 veículo) com visualização Pygame |
| `mtsp.py` | Solucionador mTSP com frota heterogênea, Pygame e integração OpenAI |
| `genetic_algorithm.py` | AG: população, fitness, crossover, mutação e ordenação |
| `draw_functions.py` | Desenho de cidades, rotas e gráfico de convergência |
| `benchmark_att48.py` | Dataset benchmark att48 (48 cidades/hospitais) |

---

## Frota Heterogênea (mtsp.py)

O mTSP modela uma frota de **carros** e **motos** com características distintas:

| Veículo | Velocidade | Autonomia |
|---------|-----------|-----------|
| 🚗 Carro | 80 km/h | 600 km |
| 🏍️ Moto | 120 km/h | 400 km |

Quando um veículo não tem autonomia suficiente para a próxima cidade, ele retorna à base para reabastecer — gerando **distância efetiva maior** que a distância direta. Esse custo de reabastecimento é incluído na função de fitness.

---

## Objetivo de Otimização

Configure a variável `OBJETIVO` para escolher a estratégia:

```python
# Opções: 'distancia' | 'tempo' | 'hibrido'
OBJETIVO = 'distancia'
ALFA_HIBRIDO = 0.5   # Só para modo híbrido: 0.0 = só distância, 1.0 = só tempo
VELOCIDADE_REF = 100 # km/h usado para normalizar tempo em km-equivalente
```

| Modo | Fitness | Comportamento |
|------|---------|--------------|
| `'distancia'` | `max(dist_ef) + peso × Σ dist_ef` | Minimiza km percorrido. Carros assumem cidades distantes (maior autonomia = menos desvios de reabastecimento). Motos ficam com cidades próximas. |
| `'tempo'` | `max(tempo) + peso × Σ tempo` | Todos os veículos terminam no mesmo horário (desvio < 2h). Motos percorrem mais km pois são mais rápidas. |
| `'hibrido'` | `(1−α) × componente_dist + α × componente_tempo` | Equilibra economia de combustível e horário de chegada. Ajuste `ALFA_HIBRIDO` conforme a necessidade. |

---

## Parâmetros Configuráveis (mtsp.py)

```python
# Algoritmo Genético
TAMANHO_POPULACAO = 100
PROBABILIDADE_MUTACAO = 0.6
GERACOES_SEM_MELHORA_PARA_PARAR = 800   # Para após 800 gerações consecutivas sem melhora
HEURISTICA = 1    # 1 = Vizinho Mais Próximo | 2 = Convex Hull
PESO_BALANCEAMENTO = 0.2                # Penalidade por desbalanceamento entre rotas

# Frota
N_CARROS = 3
N_MOTOS = 2
VELOCIDADE_CARRO = 80    # km/h
VELOCIDADE_MOTO = 120    # km/h
AUTONOMIA_CARRO = 600    # km
AUTONOMIA_MOTO = 400     # km
```

---

## Heurísticas de Inicialização

| Valor | Heurística | Descrição |
|-------|------------|-----------|
| `1` | **Vizinho Mais Próximo** | Parte do depósito e vai sempre para a cidade não visitada mais próxima |
| `2` | **Convex Hull** | Começa pela envoltória convexa e insere os pontos internos na posição de menor custo |

---

## Visualização

- **Rotas dos carros** → tons de 🟢 **verde** (escuro ao claro), gerados dinamicamente para qualquer `N_CARROS`
- **Rotas das motos** → tons de 🔴 **vermelho** (escuro ao vivo), gerados dinâmicamente para qualquer `N_MOTOS`
- **Gráfico de convergência** em tempo real com label do modo ativo (`Distância Efetiva (km)`, `Tempo (h)` ou `Híbrido (α=X)`)
- **Screenshot automático** salvo em `rotas/resultado_mtsp_<timestamp>.png`

---

## Fluxo de Execução

```
1. Gerar população inicial (heurística ou aleatória)
2. Avaliar fitness (distância efetiva / tempo / híbrido)
3. Selecionar (torneio / roleta / ranking)
4. Cruzar (OX1 – Order Crossover)
5. Mutar (inversão de segmento)
6. Elitismo: preservar melhor solução global (nunca regride)
7. Verificar convergência (N gerações sem melhora)
8. Enviar resultado ao agente ChatGPT (geração do roteiro)
```

---

## Como Usar

```bash
# Instalar dependências
pip install -r requirements.txt

# TSP (1 veículo)
python3 tsp.py

# mTSP (frota heterogênea)
python3 mtsp.py
```

- Pressione **Q** ou feche a janela para encerrar manualmente
- A melhor solução é salva em `melhor_solucao_mtsp.json`
- O roteiro final é gerado pelo agente ChatGPT e exibido no terminal

---

## Dependências

- Python 3.x
- `pygame`
- `numpy`
- `python-dotenv`
- `openai`

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` com sua chave da API:

```env
OPENAI_API_KEY=sk-...
```

---

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).