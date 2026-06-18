# Simulador FitConnect

Simulador web simples que carrega os modelos treinados do projeto e gera recomendações de posts fitness a partir das tags selecionadas pelo usuário.

É um app **independente** do dashboard do pipeline (`web/`). Sobe em outra porta (`8001`) e tem o único objetivo de demonstrar o sistema de recomendação na prática para o TCC.

## Pré-requisitos

- Ambiente `ia-recomendacao-fitness` ativado (ver [README principal](../README.md))
- Pelo menos um modelo treinado em `treinamento/modelos/<dataset_key>/<model_id>/`
- Dependências do projeto instaladas (`fastapi`, `uvicorn` já estão no `requirements.txt`)

## Como executar

A partir da **raiz do projeto**:

```bash
python -m simulador.api
```

Ou, com reload automático para desenvolvimento:

```bash
uvicorn simulador.api:app --reload --port 8001
```

Abra `http://localhost:8001` no navegador.

## Fluxo de uso

1. Selecione um modelo no dropdown (todos os modelos treinados aparecem agrupados por scale factor).
2. Selecione as tags de interesse clicando nos chips (use o campo de busca para filtrar).
3. Ajuste opções: quantidade (top-K), user ID opcional (ativa modo personalizado) e exclusão de tags exatas.
4. Clique em **Recomendar**.

### Exemplos rápidos (TCC)

No topo da página há dois exemplos pré-prontos que, com um clique, preenchem modelo, tags e opções (você ainda clica em **Recomendar** para gerar o resultado):

- **Treino de força (100% fitness)** — tags `The_New_Workout_Plan`, `Muscle_of_Love`, `The_Weight` no modelo `baseline_hibrido_padrao` (sf30). Demonstra recomendações tematicamente coerentes.
- **Fitness + termos fora do tema** — as mesmas tags + `Pizza` e `JavaScript`. Como essas duas estão fora do vocabulário fitness, o modelo as ignora (aviso "tags ignoradas") e mantém o mesmo resultado fitness — demonstra robustez a entradas fora do domínio. As tags ignoradas aparecem como chips âmbar.
- **Treino variado (novidade + recência)** — tags `The_New_Workout_Plan`, `Carry_That_Weight`, `Bicycle_Race` no modelo `baseline_hibrido_padrao` (sf30). Aqui a maioria dos posts recomendados mistura uma tag escolhida (verde = relevância) com uma tag relacionada nova, como `The_Weight` (novidade), e as datas ficam recentes (recência). Demonstra que o modelo equilibra relevância, descoberta de conteúdo relacionado e atualidade — diferente do primeiro exemplo, em que todas as tags vêm verdes.

> A primeira recomendação em cada modelo demora alguns segundos porque os artefatos (`vectorizer.pkl`, `post_matrix.npy`, `posts_cache.parquet`, etc.) são carregados em memória. Chamadas subsequentes no mesmo modelo são rápidas (cache em `treinamento.recomendar._modelos_cache`).

## Métricas de avaliação exibidas

Ao selecionar um modelo, o painel mostra as métricas de avaliação **offline** dele (as mesmas reportadas no TCC), para dar contexto sobre a qualidade do modelo:

- **NDCG@100**: mede se os itens relevantes aparecem entre os 100 primeiros e o quanto estão bem posicionados na lista. Varia de 0 a 1 (quanto maior, melhor).
- **Acerto temático@100**: proporção de consultas em que ao menos um item recomendado entre os 100 primeiros compartilha tema com o item consumido logo depois. Dois itens são do mesmo tema quando a similaridade entre suas `tags` (índice de Jaccard) é maior ou igual a 0,5. Varia de 0 a 1 (quanto maior, melhor).

> Importante: o simulador **não recalcula** essas métricas. Elas dependem de um gabarito (o que o usuário acessou depois de um instante de corte), que existe apenas na avaliação offline. Aqui, a recomendação é gerada a partir das `tags` escolhidas, sem gabarito. Por isso, os valores exibidos são os **precomputados no benchmark** (por escala da base e por modelo), apenas como referência.

## Endpoints

| Rota | Descrição |
|------|-----------|
| `GET /` | Página única do simulador |
| `GET /api/models` | Lista todos os modelos treinados disponíveis |
| `GET /api/tags?model_dir=...` | Lista as tags conhecidas pelo modelo selecionado |
| `POST /api/recommend` | Roda a recomendação (corpo JSON: `model_dir`, `tags`, `top_k`, `user_id?`, `timestamp?`, `excluir_tags_exatas?`) |

## Arquitetura

```
Browser  <-->  FastAPI (simulador/api.py, porta 8001)
                  |
                  v
              simulador/service.py
                  |
                  v
        treinamento/recomendar.py (cache de modelos)
                  |
                  v
        treinamento/modelos/<dataset_key>/<model_id>/
```

O simulador **não treina nem modifica** modelos — apenas consome os artefatos já produzidos pelo pipeline.

## Timestamp de referência

O score de recência (`time_decay`) depende de um timestamp de referência. Como o dataset LDBC SNB tem datas entre 2010–2013, usar "agora" zeraria esse sinal. O simulador, por padrão, usa o **maior timestamp do catálogo** do modelo selecionado para manter a recência ativa. É possível sobrescrever via parâmetro `timestamp` no `POST /api/recommend`.
