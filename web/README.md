# Interface Web — Dashboard do Pipeline Fitness

Interface web local que opera o pipeline de recomendacao fitness sem substituir o CLI existente (`main.py`). Funciona como camada visual sobre a logica ja implementada no projeto.

## Pre-requisitos

As dependencias do projeto base ja devem estar instaladas. Alem delas, a interface web precisa de:

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
jinja2>=3.1.0
python-multipart>=0.0.6
```

Para instalar tudo de uma vez (incluindo as dependencias do pipeline):

```bash
pip install -r requirements.txt
```

## Como executar

A partir da **raiz do projeto**:

```bash
python -m web.app
```

Ou, para desenvolvimento com auto-reload:

```bash
uvicorn web.app:app --reload --port 8000
```

Abra no navegador: **http://localhost:8000**

> O CLI continua funcionando normalmente com `python main.py`.

## O que a interface oferece

### Dashboard principal (`/`)

- **Barra de contexto** — dataset ativo, `dataset_key`, `scale_factor`, modelo alvo, familia, benchmark TCC, horario da ultima atualizacao.
- **Cards de status** — extracao, dados, splits, modelo alvo, avaliacao offline, popularidade, manual, otimizacao, benchmark TCC. Cada card mostra `OK` ou `PENDENTE`.
- **Caminhos do namespace** — `output_dir`, `dados_dir`, `splits_dir`, `models_dir`, `results_dir` (expandivel).
- **Painel de acoes** — todas as 10 opcoes do menu CLI, incluindo sub-opcoes de avaliacao e benchmark.
- **Execucao em tempo real** — barra de progresso, etapa atual, detalhe textual, console de logs expandivel.
- **Historico** — tabela com execucoes anteriores, timestamps, status e link para logs.

### Acoes disponiveis

| Acao | Endpoint |
|------|----------|
| Selecionar dataset ja baixado | `POST /api/select-dataset` |
| Baixar dataset | `POST /api/download-dataset` |
| Selecionar modelo/experimento alvo | `POST /api/select-model` |
| Configurar benchmark TCC | `POST /api/select-benchmark` |
| Rodar extracao | `POST /api/run/extraction` |
| Rodar treinamento | `POST /api/run/training` |
| Rodar avaliacao | `POST /api/run/evaluation` |
| Rodar benchmark TCC | `POST /api/run/benchmark` |
| Pipeline completo | `POST /api/run/full-pipeline` |
| Treinamento + avaliacao | `POST /api/run/training-evaluation` |
| Cancelar execucao | `POST /api/cancel` |

### API JSON

| Rota | Descricao |
|------|-----------|
| `GET /api/state` | Estado completo do pipeline |
| `GET /api/datasets` | Datasets disponiveis |
| `GET /api/models` | Modelos/experimentos do TCC |
| `GET /api/download-options` | Opcoes de download (scale factors) |
| `GET /api/eval-modes/{family}` | Modos de avaliacao por familia |
| `GET /api/execution/status` | Status da execucao atual |
| `GET /api/history` | Historico de execucoes |
| `GET /api/run/stream` | SSE — streaming de logs em tempo real |

## Arquitetura

```
Browser  <-->  FastAPI (web/app.py)  <-->  pipeline_service.py  <-->  main.py / scripts
               |  SSE streaming              |
               |  HTMX partials              |  subprocess.Popen (streaming)
               |                             |
               templates/                    .pipeline_state.json
               static/                       .execution_history.json
```

- **`pipeline_service.py`** (raiz) — camada de servico que importa funcoes de `main.py` e expoe wrappers nao-interativos (sem `input()` / `ask_yes_no()`).
- **`web/app.py`** — servidor FastAPI com rotas REST, partials HTMX e endpoint SSE.
- **`web/executor.py`** — executor de subprocessos com lock de concorrencia, streaming linha a linha e persistencia de historico.
- **`web/progress_parser.py`** — parser regex do formato `[Label] N/M (X%) - detalhe` emitido por `progress_utils.py`.
- **`web/templates/`** — templates Jinja2 (dashboard + partials atualizados via HTMX).
- **`web/static/`** — CSS (dark theme) e JS (SSE handler, acoes, notificacoes).

### Regras mantidas do pipeline

- `popularidade` e `otimizacao` so para `baseline_hibrido`
- `ltr_lightgbm` suporta apenas `offline` e `manual`
- Isolamento por `dataset_key` respeitado
- Apenas uma execucao por vez (lock de concorrencia, HTTP 409 se tentar outra)
- Estado persistido no mesmo `.pipeline_state.json` usado pelo CLI

## Estrutura de arquivos

```
web/
├── __init__.py
├── app.py                  # FastAPI app
├── executor.py             # Executor com streaming e lock
├── progress_parser.py      # Parser de linhas de progresso
├── templates/
│   ├── base.html           # Layout base
│   ├── dashboard.html      # Pagina principal
│   └── partials/
│       ├── status_header.html
│       ├── status_cards.html
│       ├── actions_panel.html
│       ├── execution_panel.html
│       └── history_panel.html
└── static/
    ├── style.css
    └── app.js
```

Arquivos gerados em tempo de execucao (ignorados pelo `.gitignore`):

- `.execution_history.json` — indice do historico de execucoes
- `.execution_logs/` — logs completos de cada execucao
