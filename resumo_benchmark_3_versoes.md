# Tabela consolidada de resultados — Benchmark TCC

Versões avaliadas: SF0.1, SF3 e SF30 do LDBC SNB (formatador `LongDateFormatter`).

Protocolo: split temporal global (70/15/15) com avaliação por interações futuras dos usuários.


## SF0.1

### Métricas de ranqueamento (top@K)

| Modelo | Queries | NDCG@5 | NDCG@10 | NDCG@20 | MRR@10 | Recall@10 | Lat. p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |
| Baseline padrao | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |
| Baseline otimizado | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |
| LTR core | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |
| LTR contextual | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |
| LTR robusto | 0 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,00 |

### Indicadores complementares

| Modelo | Cobertura | Diversidade | Novidade | Recência (dias) | Manual (critérios) |
|---|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 0,0000 | 0,0000 | 0,0000 | 0,00 | N/D |
| Baseline padrao | 0,0000 | 0,0000 | 0,0000 | 0,00 | 8/12 (66.7%) |
| Baseline otimizado | 0,0000 | 0,0000 | 0,0000 | 0,00 | 9/12 (75.0%) |
| LTR core | 0,0000 | 0,0000 | 0,0000 | 0,00 | 9/12 (75.0%) |
| LTR contextual | 0,0000 | 0,0000 | 0,0000 | 0,00 | 9/12 (75.0%) |
| LTR robusto | 0,0000 | 0,0000 | 0,0000 | 0,00 | 9/12 (75.0%) |

### Caracterização do dataset e split

- **Mensagens fitness**: 38
- **Queries candidatas**: 94 interações totais → 0 queries válidas
- **Estratégia**: temporal_global, corte temporal `cut_val_test_ms` = 1334967306374
- **Descartes por causa**:
  - `usuarios_total`: 55
  - `usuarios_menos_de_2_eventos`: 23
  - `usuarios_sem_before`: 4
  - `usuarios_sem_after`: 28
  - `future_ids_vazio`: 0
  - `ref_post_nao_resolvido`: 0
  - `ref_sem_tags`: 0
  - `queries_construidas`: 0


## SF3

### Métricas de ranqueamento (top@K)

| Modelo | Queries | NDCG@5 | NDCG@10 | NDCG@20 | MRR@10 | Recall@10 | Lat. p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 421 | 0,0000 | 0,0015 | 0,0015 | 0,0006 | 0,0048 | 3,11 |
| Baseline padrao | 421 | 0,0000 | 0,0000 | 0,0012 | 0,0000 | 0,0000 | 3,82 |
| Baseline otimizado | 421 | 0,0000 | 0,0000 | 0,0012 | 0,0000 | 0,0000 | 4,38 |
| LTR core | 421 | 0,0000 | 0,0008 | 0,0025 | 0,0003 | 0,0024 | 11,59 |
| LTR contextual | 421 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 12,04 |
| LTR robusto | 421 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 11,29 |

### Indicadores complementares

| Modelo | Cobertura | Diversidade | Novidade | Recência (dias) | Manual (critérios) |
|---|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 0,0759 | 0,0000 | 0,9887 | 256,92 | N/D |
| Baseline padrao | 0,7253 | 0,0000 | 0,9606 | 24,76 | 7/12 (58.3%) |
| Baseline otimizado | 0,4061 | 0,4949 | 0,7534 | 85,50 | 8/12 (66.7%) |
| LTR core | 0,1360 | 0,8868 | 0,8006 | 272,36 | 9/12 (75.0%) |
| LTR contextual | 0,1469 | 0,7752 | 0,9369 | 315,27 | 9/12 (75.0%) |
| LTR robusto | 0,2556 | 0,8613 | 0,9946 | 239,88 | 8/12 (66.7%) |

### Caracterização do dataset e split

- **Mensagens fitness**: 2199
- **Queries candidatas**: 10720 interações totais → 421 queries válidas
- **Estratégia**: temporal_global, corte temporal `cut_val_test_ms` = 1339842863134
- **Descartes por causa**:
  - `usuarios_total`: 6256
  - `usuarios_menos_de_2_eventos`: 3721
  - `usuarios_sem_before`: 144
  - `usuarios_sem_after`: 1969
  - `future_ids_vazio`: 1
  - `ref_post_nao_resolvido`: 0
  - `ref_sem_tags`: 0
  - `queries_construidas`: 421


## SF30

### Métricas de ranqueamento (top@K)

| Modelo | Queries | NDCG@5 | NDCG@10 | NDCG@20 | MRR@10 | Recall@10 | Lat. p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 5988 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 9,20 |
| Baseline padrao | 5988 | 0,0003 | 0,0003 | 0,0004 | 0,0003 | 0,0007 | 37,14 |
| Baseline otimizado | 5988 | 0,0004 | 0,0004 | 0,0005 | 0,0004 | 0,0007 | 37,11 |
| LTR core | 5988 | 0,0002 | 0,0003 | 0,0004 | 0,0002 | 0,0004 | 99,52 |
| LTR contextual | 5988 | 0,0003 | 0,0006 | 0,0008 | 0,0003 | 0,0014 | 100,42 |
| LTR robusto | 5988 | 0,0004 | 0,0009 | 0,0013 | 0,0006 | 0,0019 | 102,59 |

### Indicadores complementares

| Modelo | Cobertura | Diversidade | Novidade | Recência (dias) | Manual (critérios) |
|---|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 0,0104 | 0,2184 | 0,9847 | 209,59 | N/D |
| Baseline padrao | 0,4623 | 0,1181 | 0,8503 | 49,97 | 8/12 (66.7%) |
| Baseline otimizado | 0,6055 | 0,0036 | 0,9203 | 3,33 | 7/12 (58.3%) |
| LTR core | 0,3192 | 0,8287 | 0,8382 | 175,06 | 8/12 (66.7%) |
| LTR contextual | 0,3943 | 0,8060 | 0,8120 | 177,40 | 8/12 (66.7%) |
| LTR robusto | 0,3463 | 0,8332 | 0,7708 | 195,34 | 7/12 (58.3%) |

### Caracterização do dataset e split

- **Mensagens fitness**: 23485
- **Queries candidatas**: 120807 interações totais → 5988 queries válidas
- **Estratégia**: temporal_global, corte temporal `cut_val_test_ms` = 1339452740751
- **Descartes por causa**:
  - `usuarios_total`: 59419
  - `usuarios_menos_de_2_eventos`: 30381
  - `usuarios_sem_before`: 1743
  - `usuarios_sem_after`: 21306
  - `future_ids_vazio`: 1
  - `ref_post_nao_resolvido`: 0
  - `ref_sem_tags`: 0
  - `queries_construidas`: 5988
