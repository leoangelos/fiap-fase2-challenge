# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:33:42 2023

@author: SérgioPolimante
"""

## problem source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

att_48_cities_locations = [(6734, 1453),
(2233 , 10),
(5530, 1424),
 (401, 841),
(3082, 1644),
(7608, 4458),
(7573, 3716),
(7265, 1268),
(6898, 1885),
(1112, 2049),
(5468, 2606),
(5989, 2873),
(4706, 2674),
(4612, 2035),
(6347, 2683),
(6107, 669),
(7611, 5184),
(7462, 3590),
(7732, 4723),
(5900, 3561),
(4483, 3369),
(6101, 1110),
(5199, 2182),
(1633, 2809),
(4307, 2322),
 (675, 1006),
(7555, 4819),
(7541, 3981),
(3177, 756),
(7352, 4506),
(7545, 2801),
(3245, 3305),
(6426, 3173),
(4608, 1198),
 (23, 2216),
(7248, 3779),
(7762, 4595),
(7392, 2244),
(3484, 2829),
(6271, 2135),
(4985, 140),
(1916, 1569),
(7280, 4899),
(7509, 3239),
 (10, 2676),
(6807, 2993),
(5185, 3258),
(3023, 1942)]

# Nomes dos hospitais (1 por cidade, indexados de 0 a 47)
att_48_hospitals = [
    "Hospital Central São Paulo",          # 1
    "Hospital Municipal de Campinas",      # 2
    "Hospital Regional de Sorocaba",       # 3
    "Hospital Universitário de Santos",    # 4
    "Hospital Santa Casa de Ribeirão",     # 5
    "Hospital Albert Einstein",            # 6
    "Hospital Sírio-Libanês",              # 7
    "Hospital das Clínicas SP",            # 8
    "Hospital São Luiz",                   # 9
    "Hospital Municipal de Osasco",        # 10
    "Hospital Regional de Bauru",          # 11
    "Hospital Estadual de Marília",        # 12
    "Hospital Santa Marcelina",            # 13
    "Hospital São Camilo",                 # 14
    "Hospital Beneficência Portuguesa",    # 15
    "Hospital Regional de Araraquara",     # 16
    "Hospital Samaritano",                 # 17
    "Hospital Nove de Julho",              # 18
    "Hospital Oswaldo Cruz",              # 19
    "Hospital Municipal de Jundiaí",       # 20
    "Hospital Regional de Piracicaba",     # 21
    "Hospital Estadual de Franca",         # 22
    "Hospital Municipal de Limeira",       # 23
    "Hospital Santa Catarina",             # 24
    "Hospital São José",                   # 25
    "Hospital Municipal de Guarulhos",     # 26
    "Hospital Israelita",                  # 27
    "Hospital Edmundo Vasconcelos",        # 28
    "Hospital Pro Matre",                  # 29
    "Hospital Municipal de Taubaté",       # 30
    "Hospital Regional de São Carlos",     # 31
    "Hospital Municipal de Mogi",          # 32
    "Hospital Estadual de Presidente P.",  # 33
    "Hospital Municipal de Americana",     # 34
    "Hospital Regional de Registro",       # 35
    "Hospital Municipal de São Vicente",   # 36
    "Hospital Estadual de Botucatu",       # 37
    "Hospital Municipal de Assis",         # 38
    "Hospital Regional de Araçatuba",      # 39
    "Hospital São Francisco",              # 40
    "Hospital Municipal de Itapeva",       # 41
    "Hospital Estadual de Jaú",            # 42
    "Hospital Municipal de Catanduva",     # 43
    "Hospital Regional de Itapetininga",   # 44
    "Hospital Municipal de Lins",          # 45
    "Hospital Estadual de Barretos",       # 46
    "Hospital Municipal de Votuporanga",   # 47
    "Hospital Regional de Rio Claro",      # 48
]

# Prioridade de entrega por hospital (0 = Alta, 1 = Média, 2 = Baixa)
# Indexados de 0 a 47, mesma ordem que att_48_cities_locations
att_48_priorities = [
    0,  # 1  - Hospital Central São Paulo         → Alta
    1,  # 2  - Hospital Municipal de Campinas      → Média
    2,  # 3  - Hospital Regional de Sorocaba       → Baixa
    1,  # 4  - Hospital Universitário de Santos    → Média
    2,  # 5  - Hospital Santa Casa de Ribeirão     → Baixa
    0,  # 6  - Hospital Albert Einstein            → Alta
    0,  # 7  - Hospital Sírio-Libanês              → Alta
    1,  # 8  - Hospital das Clínicas SP            → Média
    0,  # 9  - Hospital São Luiz                   → Alta
    2,  # 10 - Hospital Municipal de Osasco        → Baixa
    1,  # 11 - Hospital Regional de Bauru          → Média
    2,  # 12 - Hospital Estadual de Marília        → Baixa
    1,  # 13 - Hospital Santa Marcelina            → Média
    0,  # 14 - Hospital São Camilo                 → Alta
    0,  # 15 - Hospital Beneficência Portuguesa    → Alta
    2,  # 16 - Hospital Regional de Araraquara     → Baixa
    1,  # 17 - Hospital Samaritano                 → Média
    0,  # 18 - Hospital Nove de Julho              → Alta
    1,  # 19 - Hospital Oswaldo Cruz               → Média
    2,  # 20 - Hospital Municipal de Jundiaí       → Baixa
    1,  # 21 - Hospital Regional de Piracicaba     → Média
    2,  # 22 - Hospital Estadual de Franca         → Baixa
    1,  # 23 - Hospital Municipal de Limeira       → Média
    0,  # 24 - Hospital Santa Catarina             → Alta
    2,  # 25 - Hospital São José                   → Baixa
    1,  # 26 - Hospital Municipal de Guarulhos     → Média
    0,  # 27 - Hospital Israelita                  → Alta
    1,  # 28 - Hospital Edmundo Vasconcelos        → Média
    0,  # 29 - Hospital Pro Matre                  → Alta
    2,  # 30 - Hospital Municipal de Taubaté       → Baixa
    1,  # 31 - Hospital Regional de São Carlos     → Média
    2,  # 32 - Hospital Municipal de Mogi          → Baixa
    0,  # 33 - Hospital Estadual de Presidente P.  → Alta
    1,  # 34 - Hospital Municipal de Americana     → Média
    2,  # 35 - Hospital Regional de Registro       → Baixa
    1,  # 36 - Hospital Municipal de São Vicente   → Média
    0,  # 37 - Hospital Estadual de Botucatu       → Alta
    2,  # 38 - Hospital Municipal de Assis         → Baixa
    1,  # 39 - Hospital Regional de Araçatuba      → Média
    0,  # 40 - Hospital São Francisco              → Alta
    2,  # 41 - Hospital Municipal de Itapeva       → Baixa
    1,  # 42 - Hospital Estadual de Jaú            → Média
    0,  # 43 - Hospital Municipal de Catanduva     → Alta
    2,  # 44 - Hospital Regional de Itapetininga   → Baixa
    1,  # 45 - Hospital Municipal de Lins          → Média
    0,  # 46 - Hospital Estadual de Barretos       → Alta
    1,  # 47 - Hospital Municipal de Votuporanga   → Média
    2,  # 48 - Hospital Regional de Rio Claro      → Baixa
]

# Postos de gasolina — coordenadas originais (mesma escala do benchmark att48)
# 4 pontos estrategicamente distribuídos cobrindo os 4 quadrantes do mapa
att_48_postos_gasolina = [
    (1800, 1200),   # Posto Noroeste  — cobre região esquerda/cima
    (6800, 1500),   # Posto Nordeste  — cobre região direita/cima
    (3000, 3200),   # Posto Centro-Sul — cobre região central/baixo
    (7200, 4300),   # Posto Sudeste   — cobre região direita/baixo
]

att_48_cities_order = [1,
8,
38,
31,
44,
18,
7,
28,
6,
37,
19,
27,
17,
43,
30,
36,
46,
33,
20,
47,
21,
32,
39,
48,
5,
42,
24,
10,
45,
35,
4,
26,
2,
29,
34,
41,
16,
22,
3,
23,
14,
25,
13,
11,
12,
15,
40,
9,
1,]