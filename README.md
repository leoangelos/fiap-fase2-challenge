# Solucionador TSP e mTSP com Algoritmo Genético

Este repositório contém uma implementação em Python para resolver o **Problema do Caixeiro Viajante (TSP)** e sua variante com **múltiplos veículos (mTSP)**, utilizando **Algoritmo Genético (AG)**. O objetivo é encontrar a rota mais curta que visita um conjunto de cidades exatamente uma vez e retorna à cidade de origem.

## Visão Geral

O solucionador utiliza um Algoritmo Genético para evoluir iterativamente uma população de soluções candidatas em direção a uma solução ótima ou próxima do ótimo. O AG opera simulando o processo de seleção natural, onde indivíduos com maior aptidão (menor distância de rota) têm maior probabilidade de sobreviver e gerar descendentes.

### Heurísticas de Inicialização

Ambos os arquivos (`tsp.py` e `mtsp.py`) suportam duas heurísticas para gerar a população inicial, selecionáveis pela constante `HEURISTICA`:

| Valor | Heurística | Descrição |
|-------|-----------|-----------|
| `1` | **Vizinho Mais Próximo** | A cada passo, vai para a cidade mais próxima não visitada |
| `2` | **Convex Hull** | Começa pelos pontos externos (envoltória convexa) e insere os internos na posição de menor custo |

### mTSP - Múltiplos Veículos

O `mtsp.py` estende o problema para **N veículos** partindo de um mesmo **depósito** (primeira cidade do benchmark). As cidades são distribuídas entre os veículos e cada rota é visualizada com uma cor diferente.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `tsp.py` | Solucionador TSP principal com visualização em Pygame |
| `mtsp.py` | Solucionador mTSP com múltiplos veículos e visualização multi-cores |
| `genetic_algorithm.py` | Implementação do AG: população, fitness, crossover, mutação e ordenação |
| `draw_functions.py` | Funções de desenho: cidades, rotas e gráficos usando Pygame |
| `benchmark_att48.py` | Dataset benchmark att48 com 48 cidades |

## Parâmetros Configuráveis

Os parâmetros podem ser ajustados diretamente nas constantes de cada arquivo:

```python
TAMANHO_POPULACAO = 100
PROBABILIDADE_MUTACAO = 0.5
GERACOES_SEM_MELHORA_PARA_PARAR = 800
HEURISTICA = 2  # 1 = Vizinho Mais Próximo | 2 = Convex Hull
N_VEICULOS = 3  # (apenas mtsp.py)
```

## Como Usar

```bash
# TSP - Caixeiro Viajante (1 veículo)
python3 tsp.py

# mTSP - Múltiplos Caixeiros Viajantes (N veículos)
python3 mtsp.py
```

- Pressione **Q** ou feche a janela para encerrar
- O programa para automaticamente após N gerações sem melhora
- A melhor solução é salva em arquivo JSON (`melhor_solucao.json` ou `melhor_solucao_mtsp.json`)

## Dependências

- Python 3.x
- Pygame
- NumPy
- Matplotlib

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Instâncias do Problema

- **Cidades aleatórias**: Geração aleatória de cidades
- **Problemas padrão**: Conjuntos pré-definidos com 10, 12 ou 15 cidades
- **Benchmark att48**: Dataset clássico com 48 cidades (ativo por padrão)

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

---

Sinta-se à vontade para contribuir com melhorias, correções ou novas funcionalidades. Se encontrar problemas ou tiver sugestões, abra uma issue no repositório.