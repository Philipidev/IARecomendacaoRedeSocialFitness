# IA Recomendação Rede Social Fitness

Pipeline de extração e filtragem do LDBC SNB Interactive v1 para conteúdo fitness (treinos, academia, corrida).

## Dataset necessário

O pipeline requer o **snapshot completo** do LDBC SNB (Tag, TagClass, Post, Comment, etc.).  
O arquivo `social_network-sf30-numpart-8.tar.zst` contém apenas update streams e **não funciona**.

Use um dos datasets completos do [SURF/CWI](https://repository.surfsara.nl/community/cwi):
- `social_network-sf30-CsvBasic-LongDateFormatter.tar.zst`
- `social_network-csv-basic-sf30.tar.zst`

## Estrutura

```
├── extracao_filtragem/         # Extração e filtragem
│   ├── dataset/                # Dataset bruto (.tar.zst)
│   ├── download_dataset.py     # Script de download
│   ├── pipeline.py             # Script principal
│   ├── ldbc_snb/               # Dados extraídos do .tar.zst
│   └── output/                 # Parquets gerados
│       ├── interactions_fitness.parquet
│       ├── messages_fitness.parquet
│       ├── tags_fitness.parquet
│       ├── user_interests_fitness.parquet
│       ├── user_social_graph.parquet
│       └── tag_cooccurrence.parquet
└── treinamento/                # IA de recomendação
    ├── preparacao_dados.py     # Feature engineering a partir dos parquets
    ├── treinar.py              # Treina e serializa os artefatos do modelo
    ├── recomendar.py           # Inferência — função recomendar() + CLI
    ├── dados/                  # Artefatos intermediários (gerado)
    │   ├── posts_metadata.parquet
    │   ├── interacoes_por_tag.parquet
    │   ├── social_scores.parquet
    │   ├── user_tag_profile.parquet
    │   └── tag_lista.txt
    └── modelo/                 # Artefatos do modelo treinado (gerado)
        ├── vectorizer.pkl
        ├── post_matrix.npy
        ├── tag_cooccurrence_map.pkl
        ├── popularidade.npy
        └── social_scores.npy
```

## Pré-requisitos

- Python 3.11+
- DuckDB (`pip install -r requirements.txt`)
- zstd (Linux: `apt install zstd`; Windows: [releases](https://github.com/facebook/zstd/releases))
- tar (Windows 10+ inclui)

## Execução

### 1. Baixar o dataset

```bash
pip install -r requirements.txt
python extracao_filtragem/download_dataset.py --scale-factor sf0.1
```

Scale factors: `sf0.1` (~18 MB), `sf0.3`, `sf1`, `sf3`, `sf10`, `sf30` (~20 GB).

### 2. Rodar o pipeline

```bash
python extracao_filtragem/pipeline.py --dataset-path extracao_filtragem/dataset/social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst
```

Ou, se o dataset estiver no caminho padrão após o download:

```bash
python extracao_filtragem/pipeline.py
```

### Dataset personalizado

```bash
python extracao_filtragem/pipeline.py --dataset-path caminho/para/arquivo.tar.zst
```

Ou via variável de ambiente:

```bash
export LDBC_DATASET_PATH=caminho/para/arquivo.tar.zst
python extracao_filtragem/pipeline.py
```

## Treinamento da IA de Recomendação

### 1. Preparar os dados

```bash
python treinamento/preparacao_dados.py
```

Lê os parquets de `extracao_filtragem/output/` e gera artefatos intermediários em `treinamento/dados/`, incluindo `social_scores.parquet` e `user_tag_profile.parquet`.

### 2. Dividir o dataset

```bash
python treinamento/dividir_dataset.py
```

Divide os posts em treino (70%), validação (15%) e teste (15%). Recalcula co-ocorrência e scores sociais usando apenas dados de treino.

### 3. Treinar o modelo

```bash
python treinamento/treinar.py
```

Ajusta o `MultiLabelBinarizer` sobre os nomes das tags, computa a matriz de posts e serializa os artefatos em `treinamento/modelo/`, incluindo `social_scores.npy`.

### 4. Recomendar posts (CLI)

```bash
# Listar todas as tags conhecidas pelo modelo
python treinamento/recomendar.py --listar-tags

# Recomendar posts por tags e timestamp
python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000

# Recomendar de forma personalizada para um usuário
python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000 --user-id 123

# Top 5 posts mais próximos no tempo e por tags (personalizado)
python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5 --user-id 123

# Ajustar o peso da popularidade no score padrão/fallback
python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --peso-popularidade 0.20
```

### 5. Recomendar via Python


```python
from treinamento.recomendar import recomendar

df = recomendar(
    tags=["Born_to_Run", "Superunknown"],
    timestamp=1320000000000,
    top_k=10,
    user_id=123,
)
print(df)
```

### 6. Avaliar impacto da popularidade (antes/depois)

```bash
# Avaliação real (quando houver artefatos de treino + splits)
python avaliacao/avaliar_popularidade.py --k 10 --peso-depois 0.10

# Avaliação demo (fallback sem dataset local)
python avaliacao/avaliar_popularidade.py --demo --k 10 --peso-depois 0.10
```

### Arquitetura do modelo

O score de relevância combina cinco sinais no modo padrão e cinco no modo
personalizado (`user_id` informado com perfil disponível). No modo personalizado,
a afinidade usuário-item substitui o sinal de popularidade.

| Sinal | Peso (padrão) | Peso (personalizado) | Descrição |
|---|---|---|---|
| Similaridade de conteúdo | 0.35 | 0.30 | Coseno entre vetores de tags (MultiLabelBinarizer) |
| Co-ocorrência de tags | 0.25 | 0.20 | Boost para tags relacionadas que também aparecem no post |
| Recência relativa | 0.15 | 0.15 | Decaimento exponencial pela distância em dias ao timestamp de entrada |
| Influência social | 0.15 | 0.15 | Soma do grau dos usuários que interagiram com o post no grafo social |
| Popularidade de tags | 0.10 (configurável) | - | Volume histórico de interações das tags do post (`popularidade.npy`) |
| Afinidade usuário-item | - | 0.20 | Perfil do usuário com interesses explícitos, interações recentes e sinais sociais dos vizinhos |

**Entradas:**
- `tags: List[str]` — nomes das tags (valores, não IDs)
- `timestamp: int` — timestamp em milissegundos
- `peso_popularidade: float` — ajusta o sinal de popularidade no score padrão/fallback
- `user_id: Optional[int]` — ativa recomendação personalizada quando houver perfil do usuário; caso contrário, usa o score padrão

**Saídas** (sem IDs):

| Coluna | Descrição |
|---|---|
| `message_type` | `post` ou `comment` |
| `creation_date_iso` | Data de criação (ISO 8601) |
| `tags_fitness` | Lista de tags fitness do post recomendado |
| `content_length` | Tamanho do conteúdo em caracteres |
| `language` | Idioma detectado |
| `relevance_score` | Score combinado normalizado [0, 1] |

---

## Saídas (`extracao_filtragem/output/`)

### Arquivos principais

| Arquivo | Colunas | Descrição |
|---|---|---|
| `interactions_fitness.parquet` | `user_id`, `message_id`, `event_type`, `timestamp`, `tags_fitness` | Todas as interações (like, create, reply) de usuários com conteúdo fitness |
| `messages_fitness.parquet` | `message_id`, `message_type`, `creation_date`, `content_length`, `language`, `forum_id`, `tags_fitness` | Posts e comments com pelo menos 1 tag fitness, enriquecidos com metadados de conteúdo |
| `tags_fitness.parquet` | `tag_id`, `tag_name` | Catálogo de tags fitness detectadas no dataset |

### Arquivos para treinamento da IA de recomendação

| Arquivo | Colunas | Uso na IA |
|---|---|---|
| `user_interests_fitness.parquet` | `user_id`, `tag_id`, `tag_name` | Perfil de interesse declarado do usuário — recomendação content-based ("usuário segue essa tag → mostrar posts com essa tag") |
| `user_social_graph.parquet` | `user_id`, `friend_id`, `since` | Grafo de amizades filtrado para usuários ativos em fitness — recomendação colaborativa social ("amigos de quem curtiu também curtiram") |
| `tag_cooccurrence.parquet` | `tag_a`, `tag_b`, `cooccurrences` | Co-ocorrência de tags nos mesmos posts/comments — recomendação por similaridade de tags ("quem gosta de A possivelmente gosta de B") |
