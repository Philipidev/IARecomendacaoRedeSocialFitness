"""
Gera versoes em BARRAS do comparativo de modelos do benchmark TCC, como
alternativa mais legivel ao grafico radar (gerar_radar_modelos.py).

Os valores sao os mesmos das Tabelas SF3 e SF30 do artigo (principal.tex),
fixados aqui para garantir consistencia entre figura, tabelas e texto.

As 5 dimensoes sao normalizadas por rodada em [0, 1], onde 1 = melhor da rodada.
Recencia e latencia sao invertidas (menor = melhor), de modo que, em TODAS as
dimensoes, barra mais alta = melhor desempenho.

Saidas (nao sobrescrevem o radar):
  - ArtigoTCC/figuras/comparativo_barras_v.png  (barras verticais agrupadas)
  - ArtigoTCC/figuras/comparativo_barras_h.png  (barras horizontais agrupadas)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "ArtigoTCC" / "figuras"

MODELOS = ["Popularidade", "Baseline padrão", "Baseline otimizado", "LTR (LightGBMRanker)"]
CORES = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]

# Dimensoes: (rotulo, inverter?) -- inverter=True significa "menor e melhor"
DIMENSOES = [
    ("NDCG@10", False),
    ("NDCG@100", False),
    ("Cobertura", False),
    ("Recência", True),
    ("Latência", True),
]

# Valores das Tabelas do artigo: por rodada, por modelo,
# ordem: [ndcg10, ndcg100, cobertura, recencia_dias, latencia_ms_p95]
DADOS = {
    "SF3": {
        "Popularidade": [0.0000, 0.0003, 0.080, 237.0, 8.82],
        "Baseline padrão": [0.0010, 0.0035, 0.973, 74.0, 16.53],
        "Baseline otimizado": [0.0023, 0.0032, 0.629, 6.0, 16.40],
        "LTR (LightGBMRanker)": [0.0016, 0.0082, 0.445, 151.0, 41.74],
    },
    "SF30": {
        "Popularidade": [0.0000, 0.0000, 0.047, 271.0, 25.0],
        "Baseline padrão": [0.0002, 0.0004, 0.881, 13.0, 120.9],
        "Baseline otimizado": [0.0001, 0.0002, 0.891, 5.0, 114.2],
        "LTR (LightGBMRanker)": [0.0000, 0.0000, 0.026, 152.0, 242.6],
    },
}


def normalizar_rodada(rodada: dict[str, list[float]]) -> dict[str, list[float]]:
    matriz = np.array([rodada[m] for m in MODELOS], dtype=float)  # (modelos, dims)
    norm = np.zeros_like(matriz)
    for j, (_, inverter) in enumerate(DIMENSOES):
        coluna = matriz[:, j]
        lo, hi = coluna.min(), coluna.max()
        if hi == lo:
            norm[:, j] = 1.0 if hi > 0 else 0.0
        elif inverter:
            norm[:, j] = (hi - coluna) / (hi - lo)
        else:
            norm[:, j] = (coluna - lo) / (hi - lo)
    return {m: norm[i].tolist() for i, m in enumerate(MODELOS)}


def plot_vertical() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    n_dim = len(DIMENSOES)
    n_mod = len(MODELOS)
    largura = 0.8 / n_mod
    x = np.arange(n_dim)
    rotulos_dim = [d[0] for d in DIMENSOES]

    for ax, rodada_nome in zip(axes, ("SF3", "SF30")):
        norm = normalizar_rodada(DADOS[rodada_nome])
        for i, modelo in enumerate(MODELOS):
            offset = (i - (n_mod - 1) / 2) * largura
            ax.bar(x + offset, norm[modelo], largura, label=modelo, color=CORES[i])
        ax.set_title(rodada_nome, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(rotulos_dim, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Desempenho normalizado (1 = melhor)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Comparativo dos modelos por dimensão (normalizado por rodada — maior é melhor)",
                 fontsize=12, y=1.0)

    out = FIG_DIR / "comparativo_barras_v.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def plot_horizontal() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    n_dim = len(DIMENSOES)
    n_mod = len(MODELOS)
    altura = 0.8 / n_mod
    y = np.arange(n_dim)
    rotulos_dim = [d[0] for d in DIMENSOES]

    for ax, rodada_nome in zip(axes, ("SF3", "SF30")):
        norm = normalizar_rodada(DADOS[rodada_nome])
        for i, modelo in enumerate(MODELOS):
            offset = (i - (n_mod - 1) / 2) * altura
            ax.barh(y + offset, norm[modelo], altura, label=modelo, color=CORES[i])
        ax.set_title(rodada_nome, fontsize=12, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(rotulos_dim, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[0].set_xlabel("Desempenho normalizado (1 = melhor)", fontsize=10)
    axes[1].set_xlabel("Desempenho normalizado (1 = melhor)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Comparativo dos modelos por dimensão (normalizado por rodada — maior é melhor)",
                 fontsize=12, y=1.0)

    out = FIG_DIR / "comparativo_barras_h.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    v = plot_vertical()
    h = plot_horizontal()
    print(f"Versao A (vertical)   salva em: {v}")
    print(f"Versao C (horizontal) salva em: {h}")


if __name__ == "__main__":
    main()
