# Avaliação Manual do Sistema de Recomendação

_Gerado em 2026-03-06T21:45:39.360088+00:00 (UTC)._

## Resumo Geral

- Casos executados: **3**
- Casos aprovados (todos os critérios): **0/3**

## caso_01_corrida_recente — Corrida com prioridade de recência

**Descrição:** Usuário iniciante em corrida buscando conteúdo recente e diretamente relacionado a running.  
**Entrada:** tags=['Born_to_Run', 'Running_Free'], timestamp=1320000000000, top_k=5

**Erro na execução:** `Artefato 'vectorizer.pkl' não encontrado em /workspace/IARecomendacaoRedeSocialFitness/treinamento/modelo.
Execute primeiro:
  python treinamento/treinar.py`

### Avaliação qualitativa

| Critério | Esperado | Observado | Status |
|---|---|---|---|
| aderencia_tematica | A maioria dos itens deve conter ao menos uma tag da entrada ou tag correlata de corrida. | Sem resultados para avaliar. | ❌ |
| recencia | Resultados devem concentrar posts temporalmente próximos ao timestamp informado. | Sem resultados para avaliar. | ❌ |
| variedade | Top-K não deve repetir exatamente o mesmo conjunto de tags em todos os itens. | Sem resultados para avaliar. | ❌ |
| ausencia_itens_irrelevantes | Não devem aparecer itens com tags explicitamente irrelevantes ao objetivo do usuário. | Sem resultados para avaliar. | ❌ |

## caso_02_crossfit_social — Crossfit com influência social

**Descrição:** Usuário intermediário procura conteúdos de treino funcional, aceitando variação de intensidade.  
**Entrada:** tags=['Superunknown', 'Run_with_the_Pack'], timestamp=1305000000000, top_k=7

**Erro na execução:** `Artefato 'vectorizer.pkl' não encontrado em /workspace/IARecomendacaoRedeSocialFitness/treinamento/modelo.
Execute primeiro:
  python treinamento/treinar.py`

### Avaliação qualitativa

| Critério | Esperado | Observado | Status |
|---|---|---|---|
| aderencia_tematica | Recomendações devem manter foco em treino funcional/corrida e evitar desvio temático. | Sem resultados para avaliar. | ❌ |
| recencia | Metade ou mais dos itens deve ficar dentro da janela temporal esperada. | Sem resultados para avaliar. | ❌ |
| variedade | Deve haver diversidade de combinações de tags para evitar bolha de recomendação. | Sem resultados para avaliar. | ❌ |
| ausencia_itens_irrelevantes | Itens irrelevantes devem ser residuais ou inexistentes. | Sem resultados para avaliar. | ❌ |

## caso_03_ambiguidade_tag — Entrada com tag única para avaliar diversidade

**Descrição:** Usuário avançado fornece apenas uma tag para observar expansão por co-ocorrência sem perder precisão.  
**Entrada:** tags=['Young_Hearts_Run_Free'], timestamp=1290000000000, top_k=10

**Erro na execução:** `Artefato 'vectorizer.pkl' não encontrado em /workspace/IARecomendacaoRedeSocialFitness/treinamento/modelo.
Execute primeiro:
  python treinamento/treinar.py`

### Avaliação qualitativa

| Critério | Esperado | Observado | Status |
|---|---|---|---|
| aderencia_tematica | Mesmo com entrada curta, os itens devem permanecer no domínio fitness/corrida. | Sem resultados para avaliar. | ❌ |
| recencia | Resultados recentes em relação ao timestamp devem ser favorecidos, sem zerar variedade. | Sem resultados para avaliar. | ❌ |
| variedade | Co-ocorrência deve trazer pluralidade de tags no top-K. | Sem resultados para avaliar. | ❌ |
| ausencia_itens_irrelevantes | Itens com tags bloqueadas não devem dominar o ranking. | Sem resultados para avaliar. | ❌ |

## Conclusão

Nenhum cenário foi integralmente aprovado. Revise os pesos do modelo, o dataset de treino ou ajuste os critérios esperados para refletir o comportamento desejado.
