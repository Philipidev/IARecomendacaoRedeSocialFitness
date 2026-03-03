# Treinamento — IA de Recomendação de Posts Fitness

Módulo responsável por treinar e executar a IA de recomendação de posts fitness. Consome os parquets gerados pelo pipeline de extração (`extracao_filtragem/output/`) e produz um modelo híbrido capaz de recomendar posts relevantes a partir de **nomes de tags** e um **timestamp** — sem uso de IDs.

## Estrutura

```
treinamento/
├── preparacao_dados.py     # Etapa 1: processa os parquets e gera features
├── dividir_dataset.py      # Etapa 2: divide os dados em treino/validação/teste
├── treinar.py              # Etapa 3: treina e serializa os artefatos do modelo
├── recomendar.py           # Etapa 4: inferência — função recomendar() + CLI
├── dados/                  # Artefatos intermediários (gerado automaticamente)
│   ├── posts_metadata.parquet           # Posts com apenas valores semânticos (sem IDs)
│   ├── interacoes_por_tag.parquet       # Popularidade de cada tag por volume de interações
│   ├── social_scores.parquet            # Score de influência social por post (dataset completo)
│   ├── tag_lista.txt                    # Uma tag fitness por linha (alfabético)
│   ├── event_type_lista.txt             # Tipos de evento únicos (like, create, reply)
│   ├── language_lista.txt               # Idiomas únicos presentes nos posts
│   ├── message_type_lista.txt           # Tipos de mensagem únicos (post, comment)
│   ├── user_id_lista.txt                # IDs de todos os usuários com interações fitness
│   ├── tag_cooccurrence_pares_lista.txt # Pares co-ocorrentes: tag_a|tag_b|count (desc)
│   └── splits/                          # Conjuntos divididos (gerado pelo dividir_dataset.py)
│       ├── train_posts.parquet             # 70% dos posts — usados no treino
│       ├── val_posts.parquet               # 15% dos posts — usados na validação
│       ├── test_posts.parquet              # 15% dos posts — usados no teste final
│       ├── train_interactions.parquet      # Interações dos posts de treino
│       ├── val_interactions.parquet        # Interações dos posts de validação
│       ├── test_interactions.parquet       # Interações dos posts de teste
│       ├── train_tag_cooccurrence.parquet  # Co-ocorrência recalculada só com treino
│       └── train_social_scores.parquet     # Scores sociais recalculados só com treino
└── modelo/                 # Artefatos do modelo treinado (gerado automaticamente)
    ├── vectorizer.pkl              # MultiLabelBinarizer ajustado sobre as tags
    ├── post_matrix.npy             # Matriz (n_posts × n_tags) dos vetores de posts
    ├── tag_cooccurrence_map.pkl    # Mapa {tag → [(tag_relacionada, peso)]}
    ├── popularidade.npy            # Score de popularidade normalizado por post
    └── social_scores.npy           # Score de influência social normalizado por post
```

## Pré-requisitos

```bash
pip install -r requirements.txt   # scikit-learn, pandas, numpy, pyarrow
```

O pipeline de extração deve ter sido executado antes:

```bash
python extracao_filtragem/pipeline.py
```

## Como usar

### Etapa 1 — Preparar os dados

```bash
python treinamento/preparacao_dados.py
```

Lê os parquets de `extracao_filtragem/output/` e gera os artefatos intermediários em `treinamento/dados/`:

- `posts_metadata.parquet` — posts com colunas semânticas (sem `message_id`), com `creation_date_iso` legível
- `interacoes_por_tag.parquet` — contagem de interações por tag (like + create + reply)
- `social_scores.parquet` — score de influência social por post, baseado no grau dos usuários que interagiram
- `tag_lista.txt` — uma tag fitness por linha (alfabético)
- `event_type_lista.txt` — tipos de evento únicos (like, create, reply)
- `language_lista.txt` — idiomas únicos presentes nos posts (excluindo nulos)
- `message_type_lista.txt` — tipos de mensagem únicos (post, comment)
- `user_id_lista.txt` — IDs de todos os usuários com pelo menos uma interação fitness
- `tag_cooccurrence_pares_lista.txt` — pares de tags co-ocorrentes no formato `tag_a|tag_b|cooccurrences`, ordenados por frequência decrescente

### Etapa 2 — Dividir o dataset

```bash
python treinamento/dividir_dataset.py
```

Divide os posts aleatoriamente em treino (70%), validação (15%) e teste (15%), e salva os splits em `treinamento/dados/splits/`. As proporções e a seed são configuráveis:

```bash
# Proporções diferentes
python treinamento/dividir_dataset.py --train 0.8 --val 0.1 --test 0.1

# Seed diferente
python treinamento/dividir_dataset.py --seed 123
```

### Etapa 3 — Treinar o modelo

```bash
# Treina usando apenas o conjunto de treino (recomendado)
python treinamento/treinar.py

# Treina usando o dataset completo (para versão final / produção)
python treinamento/treinar.py --dataset-completo
```

Ajusta o vetorizador, calcula a matriz de posts e salva todos os artefatos em `treinamento/modelo/`. Por padrão usa apenas os 70% de posts do split de treino. Exibe um resumo ao final:

```
Resumo dos artefatos:
  - Fonte dos dados      : split de treino
  - vectorizer.pkl       : MultiLabelBinarizer com 55 tags
  - post_matrix.npy      : 472 posts × 55 tags
  - tag_cooccurrence_map : 5 entradas
  - popularidade.npy     : 472 valores
  - social_scores.npy    : 472 valores
```

### Etapa 4 — Recomendar posts

**Via CLI:**

```bash
# Listar todas as tags conhecidas pelo modelo
python treinamento/recomendar.py --listar-tags

# Recomendar posts a partir de tags e timestamp
python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000

# Limitar a 5 recomendações
python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5

# Incluir posts com conjunto de tags idêntico à entrada
python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --incluir-exatas
```

**Via Python:**

```python
from treinamento.recomendar import recomendar

df = recomendar(
    tags=["Born_to_Run", "Superunknown"],
    timestamp=1320000000000,
    top_k=10,
)
print(df)
```

## Entradas e saídas do modelo

### Entradas

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `tags` | `List[str]` | Nomes das tags do post de referência (valores, não IDs) |
| `timestamp` | `int` | Timestamp em milissegundos do post de referência |
| `top_k` | `int` | Número de posts recomendados (padrão: 10) |

### Saída

DataFrame com os posts mais relevantes — **sem nenhum ID exposto**:

| Coluna | Descrição |
|---|---|
| `message_type` | Tipo do post: `post` ou `comment` |
| `creation_date_iso` | Data de criação no formato ISO 8601 |
| `tags_fitness` | Lista de tags fitness do post recomendado |
| `content_length` | Tamanho do conteúdo em caracteres |
| `language` | Idioma detectado |
| `relevance_score` | Score de relevância combinado, entre 0 e 1 |

## Arquitetura do modelo

O score de relevância é calculado combinando quatro sinais independentes:

```
score = 0.40 × cosine_sim + 0.25 × cooccurrence_boost + 0.15 × time_decay + 0.20 × social_influence
```

| Sinal | Peso | Como funciona |
|---|---|---|
| **Similaridade de conteúdo** | 0.40 | Coseno entre o vetor de tags da entrada e o de cada post. Posts com as mesmas tags recebem score máximo. |
| **Co-ocorrência de tags** | 0.25 | Expande as tags de entrada com tags relacionadas (do `tag_cooccurrence.parquet`) e aplica boost nos posts que as contêm. Descobre conteúdo que pessoas com gostos parecidos também curtem. |
| **Recência relativa** | 0.15 | Decaimento exponencial `exp(-0.01 × Δdias)` pela distância em dias entre o timestamp de entrada e o do post. Posts temporalmente próximos recebem maior peso. |
| **Influência social** | 0.20 | Soma do grau (número de conexões no grafo social) dos usuários que interagiram com cada post. Posts curtidos por usuários influentes (altamente conectados) recebem score mais alto. |

### Exemplo de co-ocorrência

Se a entrada for `["Young_Hearts_Run_Free"]` e o mapa de co-ocorrência indicar que essa tag aparece junto de `Superunknown` e `Run_with_the_Pack`, então posts com essas tags relacionadas também recebem boost — mesmo sem conter a tag exata da entrada.

## Fluxo completo

```
extracao_filtragem/pipeline.py
        ↓ gera parquets em output/
treinamento/preparacao_dados.py
        ↓ gera dados/ com features processadas
treinamento/dividir_dataset.py
        ↓ gera dados/splits/ com treino/validação/teste
treinamento/treinar.py
        ↓ gera modelo/ com artefatos serializados (usa split de treino)
treinamento/recomendar.py
        ↓ carrega modelo e retorna recomendações
```

---

## Conceitos essenciais

Esta seção explica em linguagem simples o que acontece por baixo dos panos. Qualquer pessoa deve conseguir entender.

### O que são treino, validação e teste?

Imagine que você está estudando para uma prova:

- **Treino (70%)** é o material de estudo — o modelo aprende os padrões com esses dados. É o maior conjunto porque quanto mais exemplos, melhor o aprendizado.
- **Validação (15%)** é o simulado — depois de estudar, você pratica com questões que ainda não viu para verificar se está indo bem. Aqui ajustamos parâmetros do modelo (como os pesos do score).
- **Teste (15%)** é a prova final — usada uma única vez, no final, para medir o desempenho real. Se você usar esses dados durante o desenvolvimento, é como ver as respostas antes da prova: o resultado será inflado e não refletirá o mundo real.

| Conjunto | Proporção | Uso |
|---|---|---|
| Treino | 70% | O modelo aprende com esses posts |
| Validação | 15% | Avalia durante o desenvolvimento; permite ajustes |
| Teste | 15% | Avaliação final — usada só uma vez |

### O que é seed?

Seed é um número que controla como o embaralhamento aleatório é feito. Com a mesma seed, o embaralhamento sempre produz o mesmo resultado — ou seja, os mesmos posts vão para treino, validação e teste, toda vez que você rodar o script.

Isso é importante para **reprodutibilidade**: qualquer pessoa que rodar o script com `--seed 42` obterá exatamente a mesma divisão.

Mudar a seed muda quais posts vão para cada grupo — o que pode ser útil para validar que os resultados não dependem de uma divisão específica.

```bash
python treinamento/dividir_dataset.py --seed 42   # divisão padrão (reproduzível)
python treinamento/dividir_dataset.py --seed 99   # divisão diferente
```

### O que é data leakage?

Data leakage ("vazamento de dados") acontece quando o modelo aprende informações do conjunto de teste durante o treinamento. Isso faz com que o modelo pareça melhor do que realmente é.

Neste projeto, tanto o `tag_cooccurrence` (relações entre tags) quanto o `social_scores` (influência do grafo social) são **recalculados usando apenas os posts de treino**. Se fossem calculados com todos os dados, o modelo aprenderia padrões que existem apenas nos posts de teste — o que seria trapacear.

```bash
# Gerados APENAS com dados de treino (sem leakage)
train_tag_cooccurrence.parquet
train_social_scores.parquet
```

### Como funciona a divisão por percentual?

Os percentuais são aplicados sobre o total de posts do dataset. Isso significa que o script funciona para qualquer tamanho de dataset, do menor (sf0.1, 674 posts) ao maior (sf30, milhões de posts):

```
n_validação = int(total × 0.15)
n_teste     = int(total × 0.15)
n_treino    = total − n_validação − n_teste  ← absorve qualquer arredondamento
```

Nenhum registro é perdido: o arredondamento vai sempre para o treino.

### Os quatro sinais do score de relevância

Quando você pede uma recomendação, o modelo combina quatro sinais para calcular o `relevance_score` de cada post:

```
score = 0.40 × similaridade_conteudo
      + 0.25 × boost_coocorrencia
      + 0.15 × fator_recencia
      + 0.20 × influencia_social
```

**1. Similaridade de conteúdo (peso 0.40 — o mais importante)**

Compara matematicamente as tags do post de entrada com as tags de cada post do catálogo. Usa o algoritmo de similaridade de cosseno: posts com as mesmas tags recebem score 1.0; posts sem nenhuma tag em comum recebem 0.0.

*Peso alto (0.40) porque relevância de conteúdo é o critério principal de recomendação.*

**2. Boost de co-ocorrência (peso 0.25 — descoberta)**

Expande a busca além das tags exatas. Se a tag `Young_Hearts_Run_Free` costuma aparecer junto de `Superunknown` nos posts, então ao buscar pela primeira, posts com a segunda também recebem um boost proporcional ao número de co-ocorrências.

*Serve para descobrir conteúdo que pessoas com gostos parecidos também curtem — mesmo que não tenha a tag exata.*

**3. Fator de recência (peso 0.15 — contexto temporal)**

Posts temporalmente próximos ao timestamp de entrada recebem pontuação maior. O decaimento segue a fórmula `exp(−0.01 × Δdias)`: posts do mesmo dia têm fator ≈ 1.0; posts de 1 ano antes/depois têm fator ≈ 0.03.

*Peso menor (0.15) porque relevância de conteúdo deve prevalecer sobre data.*

**4. Influência social (peso 0.20 — alcance no grafo)**

Cada usuário tem um **grau** no grafo social: o número de amigos que possui. Quando um usuário altamente conectado (influenciador) interage com um post, esse post provavelmente é relevante para muita gente. O sinal é pré-calculado no treino usando o `user_social_graph.parquet`, sem precisar da identidade do usuário em tempo de inferência.

*Ajuda a surfacar conteúdo popular entre pessoas bem conectadas, complementando os sinais de conteúdo.*

| Sinal | Peso | O que prioriza |
|---|---|---|
| Similaridade de conteúdo | 0.40 | Posts com as mesmas tags |
| Co-ocorrência de tags | 0.25 | Posts com tags relacionadas (descoberta) |
| Recência relativa | 0.15 | Posts temporalmente próximos |
| Influência social | 0.20 | Posts curtidos por usuários influentes |
