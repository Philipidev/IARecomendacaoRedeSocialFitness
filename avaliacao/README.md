# Avaliação Offline do Recomendador

Esta pasta executa a avaliação do modelo treinado em `treinamento/modelo/` usando os dados de teste em `treinamento/dados/splits/`.

## Comando único

```bash
python avaliacao/avaliar_modelo.py --k 5 10 20
```

> O script grava os resultados em `avaliacao/resultados/` nos formatos **JSON**, **CSV** e **Markdown**.

## Protocolo de avaliação (teste)

1. Carrega o split de teste (`test_interactions.parquet`).
2. Para cada usuário, ordena interações por tempo.
3. Cada interação vira um item de referência.
4. O ground truth é o conjunto de interações **futuras** desse mesmo usuário no teste.
5. Gera recomendações a partir das tags + timestamp do item de referência.
6. Compara Top-K recomendado com os itens reais futuros para calcular métricas.

## Métricas de ranking em K

- **Precision@K**: fração dos K recomendados que são relevantes.
- **Recall@K**: fração dos relevantes recuperados no Top-K.
- **HitRate@K**: 1 se houver ao menos um item relevante no Top-K, senão 0.
- **MAP@K**: média da precisão acumulada nas posições onde há hit, normalizada por `min(|relevantes|, K)`.
- **NDCG@K**: qualidade do ranqueamento considerando posição (ganhos descontados).

## Métricas de negócio/TCC

- **Cobertura de catálogo**: proporção de itens do catálogo que apareceram em alguma recomendação.
- **Diversidade intra-lista (tags)**: média da distância de Jaccard entre pares de itens recomendados na mesma lista.
- **Novidade (popularidade inversa)**: média de `1 / log2(freq_item + 2)` usando frequência no teste.
- **Recência média recomendada**: distância temporal média (em dias) entre item recomendado e evento de referência.

## Arquivos gerados

- `avaliacao/resultados/metricas_resumo.json`
- `avaliacao/resultados/metricas_ranking_por_k.csv`
- `avaliacao/resultados/queries_avaliadas.csv`
- `avaliacao/resultados/resumo_avaliacao.md`
