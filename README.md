# Solucionador TSP e mTSP com Algoritmo Genético

Este repositório contém uma implementação em Python para resolver o **Problema do Caixeiro Viajante (TSP)** e sua variante com **múltiplos veículos heterogêneos (mTSP)**, utilizando **Algoritmo Genético (AG)**. O objetivo é distribuir a visitação de hospitais entre uma frota mista de carros e motos, minimizando distância total ou equilibrando o tempo de chegada.

## Visão Geral

O AG evolui iterativamente uma população de soluções (permutações de cidades) usando seleção, cruzamento (OX1) e mutação. O algoritmo foi aprimorado com um **ciclo de refinamento iterativo** que aplica busca local (2-opt) e re-injeta a solução na população até a convergência total. Ao finalizar, os resultados podem ser enviados automaticamente ao **agente ChatGPT** para análise e geração do roteiro de entrega.

---

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `tsp.py` | Solucionador TSP (1 veículo) com visualização Pygame |
| `mtsp.py` | Solucionador mTSP com frota heterogênea, Pygame e integração OpenAI |
| `genetic_algorithm.py` | Funções core do AG: crossover OX1, mutação e inicialização |
| `draw_functions.py` | Desenho otimizado de cidades, rotas, feedbacks em tempo real e gráfico de convergência |
| `benchmark_att48.py` | Dataset benchmark att48 (48 hospitais) com nomes, prioridades e postos de gasolina |

---

## Frota Heterogênea (mtsp.py)

O mTSP modela uma frota de **carros** e **motos** com características distintas:

| Veículo | Velocidade | Autonomia | Capacidade de Carga |
|---------|-----------|----------|--------------------|
| 🚗 Carro | 100 km/h | 1200 km | 6 cidades |
| 🏍️ Moto | 120 km/h | 400 km | 2 cidades |

O sistema possui **duas restrições independentes** que afetam as rotas:

### Reabastecimento em Postos de Gasolina

Quando `REABASTECIMENTO_ATIVO = True`, existem **4 postos de gasolina** distribuídos estrategicamente no mapa (representados por círculos 🔵 azuis). Se o veículo não tem autonomia suficiente para a próxima cidade, ele vai até o **posto de gasolina mais próximo** e depois segue para a cidade — gerando **distância efetiva maior** que a distância direta.

**Regras:**
- O depósito **NÃO** é posto de gasolina
- Veículos podem reutilizar o mesmo posto de gasolina
- Ao retornar ao depósito por limite de carga, o tanque também é reabastecido

Se `REABASTECIMENTO_ATIVO = False`, os veículos ganham autonomia infinita.

### Capacidade de Carga

Quando `CAPACIDADE_CARGA = True`, cada veículo pode visitar um número máximo de cidades antes de **retornar ao depósito** para recarregar:

- 🚗 Carro: **6 cidades** por viagem
- 🏍️ Moto: **2 cidades** por viagem

Ao atingir o limite, o veículo volta ao depósito, recarrega (e reabastece o tanque), e segue para a próxima cidade.

Se `CAPACIDADE_CARGA = False`, não há limite de entregas por viagem.

---

## Objetivo de Otimização

Configure a variável `OBJETIVO` para escolher a estratégia:

| Modo | Fitness | Comportamento |
|------|---------|--------------|
| `'distancia'` | `max(dist_ef) + peso × Σ dist_ef` | Minimiza km percorrido (economia de combustível). A autonomia pesa nas decisões de roteamento. |
| `'tempo'` | `max(tempo) + peso × Σ tempo` | Equilibra o horário de chegada de todos os veículos, reduzindo o desvio padrão de carga de trabalho. |
| `'hibrido'` | `(1−α) × comp_dist + α × comp_tempo` | Combina economia e equilíbrio. Ajuste `ALFA_HIBRIDO` (0.0 = distância, 1.0 = tempo). |

---

## Priorização de Hospitais

Cada hospital possui um **nível de prioridade de entrega** definido no `benchmark_att48.py`:

| Nível | Valor | Visualização | Comportamento |
|-------|-------|-------------|--------------|
| Alta | `0` | 🔴 Vermelho | Urgência máxima — AG prioriza visita no início da rota |
| Média | `1` | 🟡 Amarelo | Urgência moderada — incentivo a visitar cedo |
| Baixa | `2` | 🟢 Verde | Sem urgência — livre para qualquer posição |

### Como funciona

Quando `PRIORIDADE_ATIVA = True`, a função de fitness adiciona um custo proporcional à **posição × urgência** do hospital na rota. Hospitais com alta urgência (prioridade 0) que ficam no **final** da rota aumentam muito o fitness — forçando o AG a posicioná-los no **início** de cada rota. Hospitais de baixa urgência (prioridade 2) têm peso zero e não sofrem nenhuma pressão de posicionamento.

### Parâmetros

```python
PRIORIDADE_ATIVA = True       # True = ativa penalidade por prioridade
PESO_PRIORIDADE = 50.0        # Peso da penalidade no fitness
# Quanto maior o peso, mais o AG prioriza visitar esse hospital no início da rota.
# NÃO é uma penalidade "contra" o hospital — é a urgência de visitá-lo cedo.
PESOS_NIVEL_PRIORIDADE = {
    0: 3.0,   # Alta  — urgência máxima: AG força visita no início da rota
    1: 1.0,   # Média — urgência moderada
    2: 0.0,   # Baixa — sem urgência, livre para qualquer posição
}
```

### Visualização

Quando prioridades estão ativas, as cidades no mapa Pygame são coloridas conforme seu nível de prioridade (🔴 alta, 🟡 média, 🟢 baixa). O resumo final das rotas também exibe o nome e prioridade de cada hospital visitado.

---

## Heurísticas de População Inicial

Configure `HEURISTICA` para escolher como a população inicial é gerada:

| Valor | Heurística | Descrição |
|-------|-----------|-----------|
| `1` | Vizinho Mais Próximo | Gera soluções começando de cidades diferentes, sempre indo à mais próxima |
| `2` | Convex Hull | Constrói rota pelos pontos externos e insere os internos |
| `3` | Aleatório | Permutações totalmente aleatórias (máxima diversidade genética) |

---

## Melhorias no Algoritmo Genético

A versão atual implementa táticas agressivas de mitigação de **convergência prematura**:

1. **Seleção por Torneio (k=3):** Aumenta a pressão de seleção escolhendo o indivíduo de menor custo em um grupo aleatório, substituindo a Roleta tradicional.
2. **Mutação Dinâmica (Inversão):** Opera como um 2-opt estocástico, invertendo grandes segmentos do cromossomo para forte exploração espacial.
3. **Injeção de Imigrantes (Diversidade):** A cada 50 gerações sem melhoria (estagnação), 10% da população é descartada e substituída por rotas 100% aleatórias para escapar de ótimos locais.
4. **Ciclo de Refinamento Iterativo (Pós-Otimização):**
   - Quando o AG principal converge (N gerações sem melhora), a melhor solução passa por uma Otimização Local (2-opt).
   - O 2-opt foi devidamente ajustado para otimizar a **Distância Efetiva** (considerando reabastecimentos, e não a geométrica simples).
   - Após o 2-opt, um miniciclo de AG (200 gerações) é rodado para tentar melhorar a solução gerada pelo cenário.
   - O ciclo (`2-opt -> AG -> Avalia -> 2-opt...`) continua enquanto houver melhoria real no fitness.

---

## Configurações Principais (mtsp.py)

```python
# Algoritmo Genético
TAMANHO_POPULACAO = 100
PROBABILIDADE_MUTACAO = 0.8
GERACOES_SEM_MELHORA_PARA_PARAR = 800
HEURISTICA = 2             # 1 = Vizinho Mais Próximo | 2 = Convex Hull | 3 = Aleatório

# Frota
N_CARROS = 3
N_MOTOS = 0

# Restrições dos Veículos
REABASTECIMENTO_ATIVO = True    # True = veículos precisam de combustível
CAPACIDADE_CARGA = True         # True = limite de cidades por viagem
CAPACIDADE_CARRO = 6            # Máx cidades por viagem (carro)
CAPACIDADE_MOTO = 2             # Máx cidades por viagem (moto)

# Priorização
PRIORIDADE_ATIVA = True
PESO_PRIORIDADE = 50.0

# Relatório GPT
RELATORIO_GPT = False       # True = envia resultados para o ChatGPT ao final
```

---

## Visualização (Pygame)

A UI foi fortemente otimizada para feedback em tempo real sem impacto no desempenho:
- **Painel HUD:** Overlay em tempo real com Geração atual, Valor de Fitness e Contador de Estagnação.
- **Gráfico Otimizado:** Matplotlib reutiliza a mesma figura via cache, prevenindo lentidão severa na UI visual observada em runs maiores.
- **Cidades por Prioridade:** Quando `PRIORIDADE_ATIVA = True`, as cidades são coloridas por prioridade (🔴 Alta, 🟡 Média, 🟢 Baixa). Quando desativado, todas ficam em vermelho.
- **Postos de Gasolina:** Quando `REABASTECIMENTO_ATIVO = True`, os 4 postos de gasolina são exibidos como círculos 🔵 **azuis** no mapa.
- **Rotas:** 
  - Tons de 🟢 **verde** gerados dinamicamente para carros.
  - Tons de 🔴 **vermelho** gerados dinamicamente para motos.
- **Visualização de Combustível/Carga:** Trechos em linha cheia (3px) para viagem normal, e em **linha tracejada mais fina com um marcador central** para mostrar os retornos ao depósito (recarga de carga) ou desvios aos postos de gasolina (reabastecimento).
- **Screenshot automático** de cada solução ótima em `rotas/`.

---

## Como Usar

```bash
# Instalar dependências
pip install -r requirements.txt

# Resolver TSP (1 veículo padrão)
python3 tsp.py

# Resolver mTSP (Frota Heterogênea otimizada)
python3 mtsp.py
```

- Pressione **Q** ou clique no "x" da tela Pygame para encerrar o AG prematuramente e pular para as estatísticas e ChatGPT.
- O histórico final com os resultados é exposto via Prompt gerado automaticamente ao final da execução (quando `RELATORIO_GPT = True`).

---

## Dependências

- Python 3.x
- `pygame`
- `matplotlib`
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