"""
Genera la figura de arquitectura RL del sistema IDS-DRL propuesto,
inspirada en la Figura 3 de Shaikh et al. (2025) pero adaptada al
contexto de detección de intrusiones con PPO/DQN/DDQN.

Salida: results/fig_rl_architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, 'results', 'fig_rl_architecture.png')

C_POLICY   = '#F4A580'
C_STATE    = '#A8D8E8'
C_REWARD   = '#FFE9A8'
C_INTERP   = '#C8BFE0'
C_ENV      = '#D8E8C8'
C_ACTION   = '#FFD3B6'
C_RLALGO   = '#FFFFFF'
C_BORDER   = '#333333'
C_DASHED   = '#777777'

fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

ax.add_patch(Rectangle((0.2, 3.0), 13.6, 7.8,
                       linewidth=1.2, edgecolor=C_BORDER,
                       facecolor='none'))

ax.add_patch(Rectangle((3.5, 5.0), 7.0, 5.5,
                       linewidth=1.3, edgecolor=C_DASHED,
                       facecolor='none', linestyle=(0, (5, 4))))

def rounded_box(x, y, w, h, color, label, fontsize=10, lh=1.15, weight='normal'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.18",
                         linewidth=1.1, edgecolor=C_BORDER, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center', fontsize=fontsize,
            linespacing=lh, weight=weight)

def diamond(cx, cy, w, h, color, label, fontsize=10, lh=1.15):
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.2,
                         edgecolor=C_BORDER, facecolor=color))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, linespacing=lh)

def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0), fontsize=9,
          color=C_BORDER, connectionstyle='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=18,
                                 linewidth=1.4, color=color,
                                 connectionstyle=connectionstyle))
    if label is not None:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=fontsize,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

rounded_box(5.6, 9.3, 2.8, 0.9, C_POLICY,
            "Política $\\pi_\\theta$\n(PPO · DQN · DDQN)", fontsize=10)

ax.text(8.6, 8.6, "Política\nactualizada", ha='left', va='center',
        fontsize=9, color='#444444', linespacing=1.1, style='italic')

diamond(7.0, 7.2, 3.2, 1.6, C_RLALGO,
        "Algoritmo de\nAprendizaje por\nRefuerzo", fontsize=10)

rounded_box(0.8, 7.6, 2.4, 1.2, C_STATE,
            "Observación\n(Estado)\n$S_t \\in [0,1]^{n}$", fontsize=10)

rounded_box(5.55, 5.2, 1.9, 1.0, C_REWARD,
            "Recompensa\n$R_t$", fontsize=10)

rounded_box(2.1, 5.2, 2.2, 1.0, C_INTERP,
            "Intérprete\n(recompensa\nasimétrica)", fontsize=9.5)

rounded_box(11.4, 7.6, 2.2, 1.2, C_ACTION,
            "Acción\n$A_t \\in \\{0,1\\}$\n(benigno/ataque)", fontsize=10)

rounded_box(3.6, 3.4, 7.0, 1.4, C_ENV, "", fontsize=10)
ax.text(7.1, 4.45, "Pseudo-entorno IDS",
        ha='center', va='center', fontsize=11, weight='bold')
ax.text(7.1, 3.90,
        "Datasets benchmark: NSL-KDD  ·  CICIDS2017  ·  UNSW-NB15",
        ha='center', va='center', fontsize=9.5, color='#333333')

for i, lbl in enumerate(['NSL', 'CIC', 'UNSW']):
    cx = 5.4 + i * 1.15
    ax.add_patch(FancyBboxPatch((cx - 0.35, 3.50), 0.7, 0.25,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0.8, edgecolor=C_BORDER,
                                facecolor='white'))
    ax.text(cx, 3.625, lbl, ha='center', va='center', fontsize=7)

arrow(8.4, 9.75, 11.5, 8.5, connectionstyle='arc3,rad=-0.15')

arrow(12.5, 7.6, 10.6, 4.8, connectionstyle='arc3,rad=0.15')

arrow(3.6, 4.1, 3.2, 5.2)

arrow(4.3, 5.7, 5.55, 5.7)

arrow(6.5, 6.2, 6.7, 6.5)

arrow(3.6, 4.1, 2.0, 7.6, connectionstyle='arc3,rad=-0.15')

arrow(3.2, 8.0, 5.4, 7.2)

arrow(7.0, 8.0, 7.0, 9.3)

lx, ly, lw, lh = 0.5, 0.5, 6.3, 2.0
ax.add_patch(FancyBboxPatch((lx, ly), lw, lh,
                            boxstyle="round,pad=0.04,rounding_size=0.12",
                            linewidth=0.9, edgecolor=C_BORDER,
                            facecolor='#FAFAFA'))
ax.text(lx + 0.20, ly + 1.65,
        "Función de recompensa asimétrica",
        fontsize=10, weight='bold', color='#222222', ha='left')
ax.text(lx + 0.20, ly + 1.15,
        "$R(a, y) = +1.0$  si TP        $R(a, y) = +0.1$  si TN",
        fontsize=10, ha='left')
ax.text(lx + 0.20, ly + 0.70,
        "$R(a, y) = -0.5$  si FP        $R(a, y) = -1.0$  si FN",
        fontsize=10, ha='left')
ax.text(lx + 0.20, ly + 0.20,
        "Sustento: Benaddi et al. (2022); Jacobs (2023)",
        fontsize=8.5, ha='left', style='italic', color='#555555')

nx, ny, nw, nh = 7.2, 0.5, 6.3, 2.0
ax.add_patch(FancyBboxPatch((nx, ny), nw, nh,
                            boxstyle="round,pad=0.04,rounding_size=0.12",
                            linewidth=0.9, edgecolor=C_BORDER,
                            facecolor='#FAFAFA'))
ax.text(nx + 0.20, ny + 1.65,
        "Pseudo-entorno aplicado a IDS",
        fontsize=10, weight='bold', color='#222222', ha='left')
ax.text(nx + 0.20, ny + 1.10,
        "El siguiente estado $S_{t+1}$ se obtiene por muestreo de un",
        fontsize=9.5, ha='left')
ax.text(nx + 0.20, ny + 0.75,
        "nuevo registro etiquetado del dataset; es independiente",
        fontsize=9.5, ha='left')
ax.text(nx + 0.20, ny + 0.40,
        "de la acción $A_t$ del agente.",
        fontsize=9.5, ha='left')
ax.text(nx + 0.20, ny + 0.05,
        "Sustento: Lopez-Martin et al. (2020); Caminero et al. (2019)",
        fontsize=8.5, ha='left', style='italic', color='#555555')

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] figura guardada en: {OUT_PATH}")
