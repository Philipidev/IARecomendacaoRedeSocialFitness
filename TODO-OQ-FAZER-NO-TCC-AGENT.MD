# TODO — O que fazer no TCC (handoff para outro chat)

> Arquivo de pendências para continuar o trabalho em outra conversa. Resume o que
> já foi feito, o que está pendente, e onde estão os dados gerados.

## Contexto rápido

- Artigo: [ArtigoTCC/principal.tex](ArtigoTCC/principal.tex).
- Foi adicionada uma avaliação complementar de **relevância temática** (acerto quando
  o Jaccard de `tags` do item recomendado vs. itens consumidos depois é >= 0,5).
- O código está em [avaliacao/avaliar_modelo.py](avaliacao/avaliar_modelo.py) (mudança
  aditiva: NÃO altera as métricas exatas; só acrescenta as métricas `*_tema@k`).
- Métricas temáticas geradas: `hitrate_tema@k`, `precision_tema@k`, `ndcg_tema@k` e
  `acertos_tema@k` (quantidade absoluta de consultas com >= 1 acerto temático no Top-k).
- Resultados ficam em: `avaliacao/resultados/<dataset_key>/modelos/<model_id>/offline_tema/metricas_resumo.json`
  (a pasta `offline/` original foi preservada; o `offline_tema/` é o novo).
- Limiar usado: `metadata.tema_jaccard_threshold = 0.5`. Nº de consultas: `metadata.n_queries_validas`.

`<dataset_key>`:
- SF0.1 = `social_network-sf0.1-CsvBasic-LongDateFormatter`
- SF3   = `social_network-sf3-CsvBasic-LongDateFormatter`
- SF30  = `social_network-sf30-CsvBasic-LongDateFormatter`

`<model_id>` (4 do artigo): `popularity_baseline`, `baseline_hibrido_padrao`, `baseline_hibrido_otimizado`, `ltr_lightgbm_v1_robusto`.

## O que JÁ foi feito

- Métrica temática implementada e validada (métricas exatas continuam idênticas).
- Avaliação temática rodada e ARMAZENADA para **SF0.1** e **SF3** (com `acertos_tema`).
- **SF30** temática: CONCLUÍDA e armazenada em `.../social_network-sf30-CsvBasic-LongDateFormatter/modelos/<model_id>/offline_tema/` (valores na seção abaixo).
- No `.tex` já existe a tabela `tab:relevancia_tematica` (subseção de resultados), mas
  AINDA com SF0.1 e SF3 e SEM as colunas de quantidade de Hit.

## PENDENTE (fazer)

1. ~~Confirmar a SF30 temática~~ **CONCLUÍDA** — valores já coletados (ver seção "SF30" abaixo). Armazenados em `avaliacao/resultados/social_network-sf30-CsvBasic-LongDateFormatter/modelos/<model_id>/offline_tema/metricas_resumo.json`.

2. **Atualizar a tabela `tab:relevancia_tematica` no [ArtigoTCC/principal.tex](ArtigoTCC/principal.tex):**
   - Adicionar colunas de **quantidade de Hit**: `Acertos@10 (tema)` e `Acertos@100 (tema)` (já existem `HitRate@10/@100 (tema)` e `NDCG@10 (tema)`).
   - Acrescentar as **4 linhas da SF30** (Popularidade, Baseline padrão, Baseline otimizado, LTR).
   - Conferir largura da tabela ao compilar (se estourar, usar `\resizebox{\textwidth}{!}{...}`).

3. **Figura pedida pelo professor** (comentários vermelhos ativos em
   [ArtigoTCC/principal.tex](ArtigoTCC/principal.tex) linhas ~456 e ~547):
   "ilustrar com uma figura no estilo do gráfico de *blocked cross-validation*,
   uma figura grande com tudo, COLOCAR HIT RATE, E NO SIMULADOR WEB".
   - Criar uma figura (TikZ) do protocolo de avaliação offline / split temporal,
     incluindo Hit Rate, e referenciar o simulador web.

4. **Recompilar** o `principal.tex` e revisar:
   - As 3 tabelas de resultados principais têm 11 colunas (incluem `Acertos@10`); checar se não estouram a largura.
   - Conferir a referência nova `Bergmeir2012CrossValidation` na lista de referências.

## Valores temáticos já coletados (para preencher a tabela)

Formato: `HitRate@10 (Acertos@10) | NDCG@10 | HitRate@100 (Acertos@100)`.

### SF0.1 (n = 6 consultas válidas)
| Modelo | HitRate@10 (tema) | Acertos@10 | NDCG@10 (tema) | HitRate@100 (tema) | Acertos@100 |
|---|---:|---:|---:|---:|---:|
| Popularidade | 0,000 | 0 | 0,000 | 1,000 | 6 |
| Baseline padrão | 0,000 | 0 | 0,000 | 1,000 | 6 |
| Baseline otimizado | 0,500 | 3 | 0,078 | 1,000 | 6 |
| LTR | 0,667 | 4 | 0,155 | 1,000 | 6 |

### SF3 (n = 1.663 consultas válidas)
| Modelo | HitRate@10 (tema) | Acertos@10 | NDCG@10 (tema) | HitRate@100 (tema) | Acertos@100 |
|---|---:|---:|---:|---:|---:|
| Popularidade | 0,157 | 261 | 0,087 | 0,157 | 261 |
| Baseline padrão | 0,082 | 136 | 0,067 | 0,103 | 172 |
| Baseline otimizado | 0,196 | 326 | 0,081 | 0,662 | 1101 |
| LTR | 0,182 | 302 | 0,035 | 0,500 | 832 |

### SF30 (n = 21.247 consultas válidas)
| Modelo | HitRate@10 (tema) | Acertos@10 | NDCG@10 (tema) | HitRate@100 (tema) | Acertos@100 |
|---|---:|---:|---:|---:|---:|
| Popularidade | 0,356 | 7566 | 0,121 | 0,666 | 14141 |
| Baseline padrão | 0,103 | 2179 | 0,075 | 0,116 | 2458 |
| Baseline otimizado | 0,073 | 1541 | 0,071 | 0,083 | 1768 |
| LTR | 0,267 | 5671 | 0,039 | 0,773 | 16429 |

## Observações importantes

- A métrica temática é **complementar** (não substitui as métricas exatas). Reportar o
  limiar (Jaccard >= 0,5) e a ressalva de que o domínio tem só 33 `tags`.
- NÃO re-treinar os modelos: a avaliação temática reusa os modelos já salvos em
  `treinamento/modelos/<dataset_key>/<model_id>/` e o mesmo split.
- Re-rodar a avaliação só sobrescreve `offline_tema/`; o `offline/` e o
  `benchmark_modelos.csv` consolidado não são alterados.
- Se um dia rodar o benchmark completo de novo, as métricas `*_tema@k` entram
  automaticamente no `benchmark_modelos.csv` (via `row.update(ranking)` em
  [avaliacao/benchmark_modelos.py](avaliacao/benchmark_modelos.py)).
