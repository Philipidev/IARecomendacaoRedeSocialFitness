# Extração e Filtragem — LDBC SNB Fitness

Pipeline que extrai e filtra conteúdo fitness do dataset [LDBC SNB Interactive v1](https://ldbcouncil.org/benchmarks/snb-interactive/), gerando os parquets que alimentam a IA de recomendação.

## Estrutura

```
extracao_filtragem/
├── download_dataset.py     # Baixa o dataset do repositório SURF/CWI
├── pipeline.py             # Pipeline principal de extração e filtragem
├── dataset/                # Arquivo .tar.zst bruto (baixado pelo download_dataset.py)
├── ldbc_snb/               # CSVs extraídos do .tar.zst
│   └── social_network-*/
│       ├── dynamic/        # Dados dinâmicos (person, post, comment, likes, etc.)
│       └── static/         # Dados estáticos (tag, tagclass, organisation, place, etc.)
└── output/                 # Parquets gerados pelo pipeline.py
    ├── messages_fitness.parquet
    ├── tags_fitness.parquet
    ├── tag_cooccurrence.parquet
    ├── interactions_fitness.parquet
    ├── user_interests_fitness.parquet
    └── user_social_graph.parquet
```

## Pré-requisitos

```bash
pip install -r requirements.txt   # duckdb, pyarrow, requests
```

Além disso, é necessário ter disponível no sistema:
- **zstd** — descompressão do `.tar.zst` (Linux: `apt install zstd`; Windows: [github.com/facebook/zstd/releases](https://github.com/facebook/zstd/releases))
- **tar** — incluído no Windows 10+, Linux e macOS

## Como usar

### Passo 1 — Baixar o dataset

```bash
python extracao_filtragem/download_dataset.py --scale-factor sf0.1
```

| Scale Factor | Tamanho aprox. | Uso recomendado |
|---|---|---|
| `sf0.1` | ~18 MB | Desenvolvimento e testes |
| `sf0.3` | ~50 MB | Validação local |
| `sf1` | ~160 MB | Experimentos médios |
| `sf3` | ~500 MB | Treinamento real |
| `sf10` | ~1.7 GB | Produção |
| `sf30` | ~5 GB | Produção em larga escala |

O arquivo é salvo em `extracao_filtragem/dataset/`.

> **Nota:** o repositório SURF/CWI pode colocar arquivos grandes em "tape storage". O script detecta o HTTP 409 e dispara o staging automático, aguardando até o arquivo ficar disponível.

### Passo 2 — Rodar o pipeline

```bash
# Usa o dataset padrão (sf0.1 se já baixado)
python extracao_filtragem/pipeline.py

# Ou apontando explicitamente para o arquivo
python extracao_filtragem/pipeline.py --dataset-path extracao_filtragem/dataset/social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst
```

O pipeline extrai o `.tar.zst`, descobre os CSVs, carrega no DuckDB e filtra conteúdo fitness pelas palavras-chave das TagClasses e nomes de tags.

## O que o pipeline gera (`output/`)

### Parquets principais

| Arquivo | Colunas | Descrição |
|---|---|---|
| `messages_fitness.parquet` | `message_id`, `message_type`, `creation_date`, `content_length`, `language`, `forum_id`, `tags_fitness` | Posts e comments que contêm ao menos 1 tag fitness. `tags_fitness` é uma lista de nomes de tags. |
| `tags_fitness.parquet` | `tag_id`, `tag_name` | Catálogo de todas as tags fitness identificadas no dataset. |
| `interactions_fitness.parquet` | `user_id`, `message_id`, `event_type`, `timestamp`, `tags_fitness` | Todas as interações (like, create, reply) de usuários com conteúdo fitness. |

### Parquets para treinamento da IA

| Arquivo | Colunas | Papel na IA |
|---|---|---|
| `user_interests_fitness.parquet` | `user_id`, `tag_id`, `tag_name` | Interesses declarados do usuário — base para recomendação content-based |
| `user_social_graph.parquet` | `user_id`, `friend_id`, `since` | Grafo de amizades de usuários fitness — base para recomendação colaborativa social |
| `tag_cooccurrence.parquet` | `tag_a`, `tag_b`, `cooccurrences` | Tags que aparecem juntas nos mesmos posts — base para expandir recomendações ("quem curte A pode curtir B") |

## Como o pipeline identifica conteúdo fitness

O pipeline usa duas estratégias em sequência:

1. **Estratégia A — TagClass:** busca TagClasses cujo nome contenha palavras como `sports`, `health`, `fitness`, `running`, `exercise`, `gym`, `athletic`. Todas as tags filhas dessas classes são consideradas fitness.

2. **Estratégia B (fallback) — Nome da tag:** se a estratégia A não retornar resultados, filtra diretamente pelo nome da tag com palavras como `run`, `running`, `gym`, `workout`, `crossfit`, `hiit`, `cardio`, `weight`, `sport`, entre outras.

## Lendo os parquets

```python
import pandas as pd

msgs = pd.read_parquet("extracao_filtragem/output/messages_fitness.parquet")
print(msgs.head())

cooc = pd.read_parquet("extracao_filtragem/output/tag_cooccurrence.parquet")
print(cooc)
```
