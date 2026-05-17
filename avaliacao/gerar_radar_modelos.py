"""
Gera um gráfico radar comparativo dos modelos do benchmark TCC.

Lê os resultados consolidados das rodadas SF3 e SF30 e produz uma figura com dois
radares lado a lado, cada um com 5 dimensões normalizadas em [0, 1]:

  - NDCG@10        (precisão no topo)
  - NDCG@100       (precisão em profundidade)
  - Cobertura      (fração do catálogo exposta)
  - Recência       (invertida: 1 = item mais recente; 0 = item mais antigo)
  - Latência       (invertida: 1 = mais rápido; 0 = mais lento)

A normalização é por rodada (cada eixo vai de 0 ao melhor valor observado naquela
rodada), de modo que o radar mostra o perfil relativo de cada modelo, não os
valores absolutos.

Saída: ArtigoTCC/figuras/radar_modelos.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ArtigoTCC" / "figuras" / "radar_modelos.pdf"

DATASETS = [
    ("SF3", "social_network-sf3-CsvBasic-LongDateFormatter"),
    ("SF30", "social_network-sf30-CsvBasic-LongDateFormatter"),
]

ORDEM_MODELOS = [
    "popularity_baseline",
    "baseline_hibrido_padrao",
    "baseline_hibrido_otimizado",
    "ltr_lightgbm_v1_robusto",
]

NOMES = {
    "popularity_baseline": "Popularidade",
    "baseline_hibrido_padrao": "Baseline padrão",
    "baseline_hibrido_otimizado": "Baseline otimizado",
    "ltr_lightgbm_v1_robusto": "LTR (LightGBMRanker)",
}

DIMENSOES = [
    ("NDCG@10", "ndcg10", False),
    ("NDCG@100", "ndcg100", False),
    ("Cobertura", "cobertura", False),
    ("Recência\n(inv.)", "recencia_inv", False),
    ("Latência\n(inv.)", "latencia_inv", False),
]


def carregar_metricas(dataset_key: str) -> dict[str, dict[str, float]]:
    base = ROOT / "avaliacao" / "resultados" / dataset_key / "modelos"
    out: dict[str, dict[str, float]] = {}
    for nome in ORDEM_MODELOS:
        path = base / nome / "offline" / "metricas_resumo.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        md = data["metadata"]
        r = data["ranking_metrics"]
        b = data["business_metrics"]
        out[nome] = {
            "ndcg10": float(r.get("ndcg@10", 0.0)),
            "ndcg100": float(r.get("ndcg@100", 0.0)),
            "cobertura": float(b.get("catalog_coverage", 0.0)),
            "recencia": float(b.get("avg_recommended_recency_days", 0.0)),
            "latencia": float(md.get("latencia_inferencia_ms_p95", 0.0)),
        }
    return out


def normalizar(metricas: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Normaliza cada eixo em [0, 1] onde 1 = melhor da rodada.

    Para 'recencia' e 'latencia' a normalização é invertida (menor = melhor).
    """
    if not metricas:
        return {}

    chaves = ["ndcg10", "ndcg100", "cobertura", "recencia", "latencia"]
    rangos: dict[str, tuple[float, float]] = {}
    for k in chaves:
        valores = [m[k] for m in metricas.values()]
        rangos[k] = (min(valores), max(valores))

    inv = {"recencia", "latencia"}
    norm: dict[str, dict[str, float]] = {}
    for modelo, m in metricas.items():
        norm[modelo] = {}
        for k in chaves:
            lo, hi = rangos[k]
            valor = m[k]
            if hi == lo:
                norm_val = 1.0
            elif k in inv:
                norm_val = (hi - valor) / (hi - lo)
            else:
                norm_val = (valor - lo) / (hi - lo)
            chave_dim = f"{k}_inv" if k in inv else k
            norm[modelo][chave_dim] = float(np.clip(norm_val, 0.0, 1.0))
    return norm


def plot_radar(ax, dados_normalizados: dict[str, dict[str, float]], titulo: str) -> None:
    rotulos = [d[0] for d in DIMENSOES]
    chaves = [d[1] for d in DIMENSOES]
    n = len(rotulos)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angulos += angulos[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0,25", "0,50", "0,75"], fontsize=7, color="gray")
    ax.grid(True, linestyle=":", alpha=0.5)

    cores = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
    for cor, modelo in zip(cores, ORDEM_MODELOS):
        if modelo not in dados_normalizados:
            continue
        valores = [dados_normalizados[modelo].get(k, 0.0) for k in chaves]
        valores += valores[:1]
        ax.plot(angulos, valores, color=cor, linewidth=1.6, label=NOMES[modelo])
        ax.fill(angulos, valores, color=cor, alpha=0.10)

    ax.set_title(titulo, fontsize=11, pad=18, fontweight="bold")


def main() -> None:
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(11, 5.4), subplot_kw={"projection": "polar"}
    )

    legendas_ax = None
    for ax, (label, key) in zip(axes, DATASETS):
        metricas = carregar_metricas(key)
        if not metricas:
            ax.set_title(f"{label} (sem dados)", fontsize=11)
            continue
        normalizadas = normalizar(metricas)
        plot_radar(ax, normalizadas, label)
        legendas_ax = ax

    if legendas_ax is not None:
        handles, labels = legendas_ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(
        "Perfil comparativo dos modelos por dimensão (normalizado por rodada)",
        fontsize=12,
        y=1.02,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Figura salva em: {OUT}")
    print(f"PNG  salvo em: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
