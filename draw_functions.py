# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:03:11 2023

@author: SérgioPolimante
"""
import pylab
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import pygame
from typing import List, Tuple

matplotlib.use("Agg")

_plot_cache = {}  # Armazena figura/eixos entre chamadas para evitar recriação a cada frame

def draw_plot(screen: pygame.Surface, x: list, y: list, x_label: str = 'Generation', y_label: str = 'Fitness') -> None:
    """
    Desenha gráfico de convergência na tela Pygame.
    Reutiliza a figura Matplotlib entre chamadas para melhor performance
    (evita recriar figura/canvas a cada geração).
    """
    global _plot_cache

    cache_key = id(screen)
    if cache_key not in _plot_cache:
        # Cria figura e canvas apenas uma vez
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        line, = ax.plot([], [], color='steelblue')
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        plt.tight_layout()
        canvas = FigureCanvasAgg(fig)
        _plot_cache[cache_key] = {'fig': fig, 'ax': ax, 'line': line, 'canvas': canvas}

    cache = _plot_cache[cache_key]
    ax = cache['ax']
    line = cache['line']
    canvas = cache['canvas']

    # Atualiza apenas os dados da linha (sem recriar a figura)
    line.set_xdata(x)
    line.set_ydata(y)
    ax.set_ylabel(y_label)

    if len(x) > 1:
        ax.set_xlim(0, max(x))
    if len(y) > 1:
        margin = (max(y) - min(y)) * 0.1 or 1
        ax.set_ylim(min(y) - margin, max(y) + margin)

    canvas.draw()
    raw_data = bytes(canvas.buffer_rgba())
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    screen.blit(surf, (0, 0))

def draw_cities(screen: pygame.Surface, cities_locations: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], node_radius: int) -> None:
    """
    Draws circles representing cities on the given Pygame screen.
    """
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)



def draw_paths(screen: pygame.Surface, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1):
    """
    Draw a path on a Pygame screen.
    """
    pygame.draw.lines(screen, rgb_color, True, path, width=width)


def draw_text(screen: pygame.Surface, text: str, color: pygame.Color) -> None:
    """
    Draw text on a Pygame screen.
    """
    pygame.font.init()

    font_size = 15
    my_font = pygame.font.SysFont('Arial', font_size)
    text_surface = my_font.render(text, False, color)

    cities_locations = []  # Assuming you have this list defined somewhere
    text_position = (np.average(np.array(cities_locations)[:, 0]), HEIGHT - 1.5 * font_size)

    screen.blit(text_surface, text_position)


def draw_info_overlay(screen: pygame.Surface, geracao: int, fitness: float, sem_melhora: int, n_veiculos: int) -> None:
    """
    Desenha overlay de texto com informações da geração atual.
    Exibe geração, fitness e contagem de estagnação para feedback visual em tempo real.
    """
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 13)
    cor = (20, 20, 80)

    linhas = [
        f"Gen: {geracao}",
        f"Fitness: {round(fitness, 1)}",
        f"Estag: {sem_melhora}",
        f"Veic: {n_veiculos}",
    ]
    x_base = 5
    y_base = 5
    for i, linha in enumerate(linhas):
        surf = font.render(linha, True, cor)
        screen.blit(surf, (x_base, y_base + i * 16))
