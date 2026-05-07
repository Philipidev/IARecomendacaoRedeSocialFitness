# Guia das Métricas de Avaliação do Recomendador

> Este documento explica, em linguagem acessível, **todas as métricas** usadas
> para avaliar o sistema de recomendação do TCC. O objetivo é que qualquer
> pessoa — inclusive sem fundo em recuperação de informação ou aprendizado de
> máquina — entenda **o que cada métrica mede, como é calculada, por que é
> relevante e como ler os números**.
>
> Use este guia como base para a discussão de métricas no texto do TCC. A
> ordem das seções vai do mais geral ao mais específico, e termina com
> recomendações práticas de como apresentar tudo na monografia.

---

## Sumário

1. [Como o recomendador é avaliado](#1-como-o-recomendador-é-avaliado)
2. [O que significa "@K" (arroba K)](#2-o-que-significa-k-arroba-k)
3. [Métricas de **ranking** (centro do TCC)](#3-métricas-de-ranking-centro-do-tcc)
   - [3.1. Precision@K](#31-precisionk--precisão)
   - [3.2. Recall@K](#32-recallk--abrangência)
   - [3.3. Hit Rate (HR@K)](#33-hit-rate-hrk--taxa-de-acerto)
   - [3.4. MAP@K](#34-mapk--mean-average-precision)
   - [3.5. NDCG@K](#35-ndcgk--normalized-discounted-cumulative-gain)
   - [3.6. MRR@K](#36-mrrk--mean-reciprocal-rank)
4. [Métricas de **negócio / experiência**](#4-métricas-de-negócio--experiência)
   - [Cobertura de catálogo](#41-cobertura-de-catálogo-coverage)
   - [Diversidade intra-lista](#42-diversidade-intra-lista-diversity)
   - [Novidade](#43-novidade-novelty)
   - [Recência média](#44-recência-média-recommended-recency)
   - [Latência](#45-latência-de-inferência)
5. [Intervalos de confiança por bootstrap](#5-intervalos-de-confiança-por-bootstrap)
6. [Como interpretar valores absolutos](#6-como-interpretar-valores-absolutos)
7. [Quais escolher para o TCC](#7-quais-escolher-para-o-tcc)
8. [Glossário rápido](#8-glossário-rápido)

---

## 1. Como o recomendador é avaliado

A avaliação **offline** funciona assim, em três passos:

1. **O protocolo monta queries de teste.** Para cada usuário, escolhemos uma
   interação como "ponto de referência" e definimos como **gabarito**
   (`future_ids`) o conjunto de itens que o usuário consumiu **depois** dessa
   referência.

2. **O modelo gera uma lista ordenada de K recomendações** (`Top-K`). Ele só
   conhece a referência (tags, timestamp), não o futuro do usuário.

3. **Comparamos a lista com o gabarito.** As métricas medem, de ângulos
   diferentes, o quanto a lista do modelo se parece com o que o usuário
   realmente consumiu.

> **Por que é difícil acertar?** O catálogo costuma ter milhares de itens; o
> gabarito tem poucos. Acertar exatamente os próximos itens consumidos é, por
> natureza, uma tarefa **muito esparsa** — esse é o motivo de números baixos
> serem normais em recsys (veja a [seção 6](#6-como-interpretar-valores-absolutos)).

---

## 2. O que significa "@K" (arroba K)

O sufixo `@K` indica que a métrica foi calculada considerando **apenas as K
primeiras posições da lista recomendada**. Por exemplo:

- `Precision@5` → olha só os 5 primeiros itens recomendados.
- `NDCG@10` → olha só os 10 primeiros itens recomendados.
- `Recall@20` → olha só os 20 primeiros itens recomendados.

### Por que K = 10 (o "padrão da indústria")?

Porque, em interfaces reais (feed do Instagram, página de busca do Spotify, página inicial
do Netflix), o usuário tipicamente vê de 5 a 10 itens **sem precisar rolar a tela**.
Métricas no top-10 refletem a experiência real. Reportar K = 5, 10 e 20 dá uma
visão completa: o modelo é bom no topo (5), no contexto comum (10) e numa lista
expandida (20)?

> **Regra prática:** quanto **menor** o K, mais **rigorosa** a métrica. Acertar
> no Top-5 é mais difícil que acertar no Top-20.

---

## 3. Métricas de **ranking** (centro do TCC)

Todas as métricas abaixo variam de **0 (péssimo) a 1 (perfeito)**.

Para os exemplos, suponha:

- O modelo recomendou estes 10 itens, em ordem:
  `[A, B, C, D, E, F, G, H, I, J]`
- O **gabarito** do usuário é: `{C, F, X, Y}` (ou seja, ele consumiu, depois da
  referência: C, F, X e Y, em alguma ordem).
- Acertos no Top-10: **C** (posição 3) e **F** (posição 6). Total = 2 acertos.

---

### 3.1. Precision@K — "Precisão"

**Pergunta que responde:** dos K itens que recomendei, **qual fração** era
relevante?

**Fórmula:**
```
Precision@K = (nº de acertos no Top-K) / K
```

**Exemplo:** `Precision@10 = 2 / 10 = 0,20 (20%)`.

**Quando usar:** você quer medir o quanto a lista é "limpa" — sem ruído. Útil
em cenários onde **mostrar item irrelevante incomoda o usuário** (anúncios,
e-commerce caro).

**Limitação:** ignora **quantos relevantes existiam fora da lista**. Se o
gabarito tem 100 itens e você acertou 2 no Top-10, Precision@10 = 0,20 — mesmo
que tenha deixado 98 relevantes de fora.

**Como ler valores:**
| Faixa | Interpretação |
|---|---|
| 0,00 | Nenhum acerto no Top-K (ruído total). |
| 0,10 | 1 a cada 10 itens é relevante — comum em recsys reais. |
| 0,30 | Lista bem afiada — bom resultado para domínios esparsos. |
| 0,50+ | Excelente, raramente atingido fora de domínios densos. |

---

### 3.2. Recall@K — "Abrangência"

**Pergunta que responde:** dos itens relevantes que o usuário **realmente
consumiu**, **qual fração** consegui colocar nas K primeiras posições?

**Fórmula:**
```
Recall@K = (nº de acertos no Top-K) / (nº total de relevantes no gabarito)
```

**Exemplo:** o gabarito tem 4 itens (`C, F, X, Y`). Acertos no Top-10 = 2.
`Recall@10 = 2 / 4 = 0,50 (50%)`.

**Quando usar:** você quer medir o quanto **não está deixando relevantes para
trás**. Crítico em sistemas onde "perder uma recomendação boa" tem custo
(saúde, busca jurídica, recomendação científica).

**Relação com Precision:** Precision e Recall são **complementares** —
melhorar uma costuma piorar a outra. Por isso reportamos ambas.

**Como ler valores:**
| Faixa | Interpretação |
|---|---|
| 0,00 | O modelo não recuperou nenhum relevante no Top-K. |
| 0,10 | 10% dos relevantes foram pegos — modesto. |
| 0,30 | 30% — bom em catálogo grande. |
| 0,50+ | Metade ou mais — excelente. |
| 1,00 | Pegou todos os relevantes (raro, exceto se o gabarito for pequeno). |

---

### 3.3. Hit Rate (HR@K) — "Taxa de acerto"

**Pergunta que responde:** entre as K recomendações, **pelo menos um** item
era relevante?

**Fórmula:**
```
HR@K = 1 se há ≥ 1 acerto no Top-K
HR@K = 0 caso contrário
```

E a média sobre todas as queries.

**Exemplo:** `HR@10 = 1` na nossa lista, porque há pelo menos um acerto (C, na
posição 3).

**Quando usar:** é a métrica mais **permissiva** e a mais intuitiva — útil
para responder "**fizemos pelo menos uma boa sugestão?**". Em recsys com
gabarito muito esparso (1-2 itens por query), HR é frequentemente o sinal
mais estável.

**Como ler valores:**
| Faixa | Interpretação |
|---|---|
| 0,00 | Em **0%** das queries, nenhum item do Top-K era relevante. |
| 0,10 | Em **10%** das queries, pelo menos um acerto — fraco. |
| 0,30 | Em **30%** das queries — competitivo em domínios esparsos. |
| 0,50+ | Em **metade** ou mais — muito bom. |

> **Observação:** quando o gabarito tem **só 1 item**, `HR@K = Recall@K`. Por
> isso muitos artigos de recsys reportam só HR.

---

### 3.4. MAP@K — *Mean Average Precision*

**Pergunta que responde:** combina **precisão** com a **posição dos acertos**,
premiando acertos mais cedo na lista.

**Fórmula (intuição):**
1. Para cada acerto na posição `i`, calculamos a Precision@i (a precisão
   considerando só as primeiras `i` posições).
2. Somamos essas precisões e dividimos pelo número total de relevantes (até K).
3. Tiramos a média sobre todas as queries.

**Exemplo:**
- Acerto na posição 3 → P@3 = 1/3 = 0,333
- Acerto na posição 6 → P@6 = 2/6 = 0,333
- AP = (0,333 + 0,333) / min(4, 10) = 0,666 / 4 = **0,167**

**Quando usar:** quando a lista pode ter vários relevantes e você quer que os
melhores apareçam **no topo**, não espalhados pelo Top-K.

**Como ler valores:** mesma faixa de Precision, com viés a recompensar
recuperação cedo na lista.

---

### 3.5. NDCG@K — *Normalized Discounted Cumulative Gain*

A **métrica oficial** do TCC (alvo do `metric_target` no benchmark).

**Pergunta que responde:** o quanto os itens relevantes estão **concentrados
no topo** da lista?

**Fórmula (intuição):**
1. Para cada posição `i` no Top-K, atribua um "ganho":
   - 1 se a posição contém um item relevante
   - 0 caso contrário
2. **Desconte** o ganho pela posição: ganho / log₂(i + 1). Isto é, posições
   mais profundas valem **menos**.
3. Some todos os ganhos descontados → **DCG**.
4. Calcule o **DCG ideal** (IDCG): se todos os relevantes estivessem nas
   primeiras posições.
5. **NDCG = DCG / IDCG**, normalizado entre 0 e 1.

**Exemplo:**
Para a lista `[A, B, C(✓), D, E, F(✓), G, H, I, J]` com 4 relevantes no gabarito:
- Acerto na posição 3 → ganho descontado = 1 / log₂(4) = 0,500
- Acerto na posição 6 → ganho descontado = 1 / log₂(7) = 0,356
- **DCG** = 0,500 + 0,356 = 0,856
- **IDCG** (4 relevantes nas posições 1, 2, 3, 4):
  1/log₂(2) + 1/log₂(3) + 1/log₂(4) + 1/log₂(5) = 1,000 + 0,631 + 0,500 + 0,431 = 2,562
- **NDCG@10** = 0,856 / 2,562 ≈ **0,334**

**Por que NDCG é tão usada:**
- **Sensível à posição:** acertar na posição 1 vale mais que acertar na 10.
- **Normalizada:** sempre entre 0 e 1, comparável entre datasets.
- **Suporta relevância graduada** (1, 2, 3 estrelas), embora aqui usemos
  binária (relevante / não relevante).

**Como ler valores:**
| Faixa | Interpretação |
|---|---|
| 0,000 | Nenhum acerto no Top-K. |
| 0,001 – 0,01 | Pontuação muito baixa — comum em datasets pequenos ou esparsos. |
| 0,05 – 0,15 | Faixa típica de baselines em recsys "no mundo real". |
| 0,15 – 0,30 | Bom modelo. |
| 0,30+ | Excelente — frequente em domínios densos (música, vídeo). |

> **Por que o paper de SF3 mostrou NDCG@10 = 0,0009?** A combinação de split
> errado + gabarito reduzido + catálogo grande forçou matematicamente o número
> a ser quase zero. Veja [seção 6](#6-como-interpretar-valores-absolutos).

---

### 3.6. MRR@K — *Mean Reciprocal Rank*

**Pergunta que responde:** **quão cedo** apareceu o **primeiro** acerto?

**Fórmula:**
```
MRR@K = 1 / posição_do_primeiro_acerto
```

E média sobre todas as queries.

**Exemplo:** na nossa lista, o primeiro acerto é C, na posição 3.
`MRR@10 = 1 / 3 ≈ 0,333`.

**Quando usar:** quando o objetivo é **achar uma boa recomendação rápido**,
sem se preocupar com as outras (busca em motores, FAQ, "I'm feeling lucky").

**Como ler valores:**
| Valor | Interpretação |
|---|---|
| 1,00 | Primeiro acerto sempre na posição 1. |
| 0,50 | Em média, primeiro acerto na posição 2. |
| 0,33 | Em média, posição 3. |
| 0,10 | Em média, posição 10. |
| 0,00 | Nunca houve acerto. |

---

### Resumo das métricas de ranking

| Métrica | O que premia | Quando ela "brilha" |
|---|---|---|
| **Precision@K** | Lista limpa | Quando irrelevante = custo (anúncio caro) |
| **Recall@K** | Cobrir todos os relevantes | Quando perder um = custo (busca jurídica) |
| **HR@K** | "Pelo menos um acerto" | Comunicação simples; gabarito pequeno |
| **MAP@K** | Precisão **e** posição | Listas com vários relevantes |
| **NDCG@K** | Posição (com desconto suave) | **Padrão da literatura de ranking** |
| **MRR@K** | O **primeiro** acerto | Busca, perguntas-respostas |

---

## 4. Métricas de **negócio / experiência**

Métricas de ranking dizem se o modelo "acertou", mas não toda a história. Um
modelo pode ter NDCG alto recomendando sempre os 10 mesmos itens populares —
isso é **ruim para o negócio** (cobertura zero, sem novidade). As métricas de
negócio capturam essas dimensões.

### 4.1. Cobertura de catálogo (*coverage*)

**O que mede:** **qual fração** do catálogo total é exibida ao menos uma vez ao
longo de todas as recomendações da avaliação.

**Fórmula:**
```
Cobertura = (itens únicos recomendados) / (tamanho do catálogo)
```

**Faixa:** 0 a 1.

**Por que importa:**
- Cobertura 0,05 → o modelo só sabe recomendar 5% do catálogo. Itens novos ou
  de cauda longa nunca aparecem.
- Cobertura 0,80 → o modelo explora bem o catálogo (bom para descoberta e para
  produtores de conteúdo da plataforma).

**Trade-off:** modelos de Popularidade tipicamente têm **cobertura baixa**
(2-10%), porque sempre recomendam os mesmos itens. Modelos personalizados
costumam ter cobertura alta.

---

### 4.2. Diversidade intra-lista (*diversity*)

**O que mede:** **dentro de uma lista**, quão **diferentes** são os itens entre
si (em termos de tags). Calculada como `1 − média(Jaccard de tags entre pares)`.

**Faixa:** 0 (todos idênticos) a 1 (totalmente disjuntos).

**Por que importa:** uma lista com 10 itens **sobre o mesmo assunto** pode
parecer redundante. Diversidade alta sugere uma lista **temática variada**,
mas pode também indicar que o modelo está "chutando" sem foco.

**Como ler:**
| Valor | Interpretação |
|---|---|
| 0,2 – 0,4 | Lista coesa (todos parecidos). |
| 0,5 – 0,7 | Equilíbrio — varia mas tem foco. |
| 0,8+ | Lista muito variada — pode ser "perdida" se a query é específica. |

---

### 4.3. Novidade (*novelty*)

**O que mede:** o quanto o modelo recomenda itens **menos populares**.
Calculada como `1 / log₂(frequência + 2)` — itens muito populares dão pontuação
baixa; itens raros dão pontuação alta.

**Faixa:** 0 a 1 (limites teóricos).

**Por que importa:** combate o efeito **filter bubble** e o **cold-start de
itens novos**. Um modelo com novidade alta ajuda usuários a **descobrir**
conteúdo, em vez de só reforçar o que todo mundo já vê.

---

### 4.4. Recência média (*recommended recency*)

**O que mede:** a **distância temporal média**, em **dias**, entre a data de
criação dos itens recomendados e o timestamp da query.

**Diferentemente das outras**, esta métrica **não** é entre 0 e 1 — é uma
contagem de dias.

**Por que importa:**
- Em redes sociais, conteúdo **fresco** engaja mais.
- Recência alta (>180 dias) sugere que o modelo está recomendando posts antigos.
- Recência baixa (<30 dias) é um forte sinal positivo em domínios como notícias,
  fitness, moda, esportes.

---

### 4.5. Latência de inferência

**O que mede:** o tempo, em milissegundos, para o modelo **gerar uma
recomendação** após receber a query. Reportamos:
- **p50** (mediana): o tempo típico.
- **p95** (percentil 95): o tempo nos 5% piores casos.

**Por que importa:** em produção, latência > 200 ms degrada a experiência do
usuário. Modelos como **LightGBM** são intrinsecamente mais lentos que
heurísticas (precisam fazer N predições por candidato), mas costumam ficar
abaixo de 50 ms em catálogos da escala do TCC.

---

## 5. Intervalos de confiança por bootstrap

Quando você reporta `NDCG@10 = 0,082`, esse é só **um número**. Mas se o
experimento for refeito com queries amostradas diferentes, daria 0,07 ou 0,09?
O **intervalo de confiança (IC) bootstrap 95%** responde:

> Estamos 95% confiantes de que o **valor verdadeiro** da métrica está entre
> X e Y.

**Como funciona o bootstrap (intuição):**
1. Você tem N queries de teste, cada uma com uma medida de NDCG@10.
2. **Reamostra com reposição:** sorteia N queries (algumas podem repetir, outras
   sumir) e calcula a média de NDCG@10.
3. Repete isso 1000 vezes → você tem 1000 médias possíveis.
4. O **percentil 2,5%** e o **percentil 97,5%** são os limites do IC 95%.

**Como ler na sua tese:**
- `NDCG@10 = 0,082 (IC 95%: 0,061 – 0,103, n=367)` → o efeito é **estatisticamente
  presente** (intervalo não cruza zero).
- `NDCG@10 = 0,005 (IC 95%: 0,000 – 0,012, n=5)` → resultado **inconclusivo**:
  o intervalo cruza zero. Não dá para garantir que o modelo é melhor que o
  acaso.

**Por que reportar IC?**
- A banca **vai notar**. Reportar só a média sem IC é amador.
- IC é a maneira correta de **comparar dois modelos**: se os intervalos não
  se sobrepõem, a diferença é significativa.

---

## 6. Como interpretar valores absolutos

> **A pergunta que mais aparece em defesas:** "0,01 é bom ou ruim?"
> A resposta: **depende** — do dataset, do baseline e da estratégia de avaliação.

### Por que recsys tem números **baixos** naturalmente

Compare com **classificação** (acurácia):
- Em classificação binária, 50% é o "chute". 90% é considerado bom, 99% é
  excelente.
- Em recsys, o "chute aleatório" costuma dar valores muito menores, porque o
  modelo precisa **escolher 10 itens entre milhares**, e o gabarito tem só
  poucos itens.

**Conta de padaria:**
- Catálogo com 10 000 itens, gabarito com 5 itens.
- Probabilidade aleatória de o Top-10 conter ao menos um relevante:
  `1 − (9995/10000)^10 ≈ 0,005`.
- Ou seja, mesmo o **acaso** dá ~0,5% de hit rate.

Por isso, **NDCG@10 de 0,05** já pode ser **um modelo competente** em
domínios com catálogo grande.

### Faixas típicas em datasets reais (para comparação)

Os números abaixo são ordens de grandeza tipicamente observadas na
literatura de recsys (MovieLens, Amazon Reviews, RetailRocket, Last.fm):

| Métrica @10 | Baseline (Popularidade) | Modelo bom (BPR, ItemKNN) | Estado da arte (Transformers) |
|---|---|---|---|
| HR@10 | 0,03 – 0,10 | 0,10 – 0,30 | 0,30 – 0,55 |
| NDCG@10 | 0,02 – 0,06 | 0,05 – 0,15 | 0,15 – 0,35 |
| MRR@10 | 0,01 – 0,04 | 0,03 – 0,10 | 0,10 – 0,25 |
| Recall@10 | 0,03 – 0,10 | 0,10 – 0,30 | 0,30 – 0,55 |
| Coverage | 0,02 – 0,10 | 0,30 – 0,70 | 0,40 – 0,80 |

> **Não compare com classificação.** Quem espera ver 0,8 ou 0,9 em recsys
> está esperando errado.

### Como saber se 0,01 é bom ou ruim

**Sempre compare contra um baseline.** Se o seu modelo dá `NDCG@10 = 0,01` mas
o baseline aleatório dá `0,002`, o modelo é **5× melhor que o acaso** —
significativo. Se os dois dão 0,01, o modelo aprendeu **nada**.

Por isso o TCC inclui:
- **Baseline híbrido manual** (com pesos da literatura).
- **Baseline híbrido otimizado** (pesos por grid search no validation).
- **Popularidade pura** (referência mínima — se o modelo perder para isso,
  algo está errado).
- **LTR (LightGBM)** com 3 configurações.

A **história do TCC** não é "consegui NDCG = X". É: **modelo Y supera o
modelo Z em W%, com IC 95% que não se sobrepõe**.

---

## 7. Quais escolher para o TCC

### Métricas **obrigatórias** (sempre reportar)

| Métrica | Por quê |
|---|---|
| **NDCG@10** | Métrica oficial; sensível a posição. |
| **HR@10** | Intuitiva; estável quando gabarito é pequeno. |
| **MRR@10** | Premia "pelo menos uma boa recomendação rápido". |
| **Recall@10** | Cobertura do gabarito. |
| **Cobertura de catálogo** | Mostra que o modelo não é míope. |
| **Latência p95** | Viabilidade prática. |

### Métricas **complementares** (recomendado)

- **Precision@10**, **MAP@10** — para análise mais fina.
- **NDCG@5** e **NDCG@20** — sensibilidade ao tamanho da janela.
- **Diversidade**, **Novidade**, **Recência** — para a discussão qualitativa.

### Como apresentar

**Tabela principal de resultados** (uma por scale factor — SF0.1, SF1, SF3, SF30):

| Modelo | Queries | NDCG@10 | IC 95% | HR@10 | IC 95% | MRR@10 | Latência p95 |
|---|---:|---:|---|---:|---|---:|---:|
| Popularidade | 367 | 0,021 | [0,015 – 0,027] | 0,074 | [0,054 – 0,094] | 0,012 | 1 ms |
| Baseline padrão | 367 | 0,055 | [0,043 – 0,067] | 0,182 | [0,151 – 0,213] | 0,031 | 4 ms |
| Baseline otimizado | 367 | 0,082 | [0,068 – 0,096] | 0,236 | [0,201 – 0,271] | 0,047 | 4 ms |
| LTR core | 367 | 0,098 | [0,082 – 0,114] | 0,278 | [0,243 – 0,313] | 0,055 | 14 ms |

**Como descrever os resultados (modelo de texto pronto):**

> O modelo **LTR core** atingiu NDCG@10 = 0,098 (IC 95%: 0,082 – 0,114), o
> melhor resultado entre as configurações testadas. Esse valor representa um
> **ganho relativo de 19,5% sobre o baseline otimizado** (0,082) e um ganho
> de **4,7×** sobre a referência de Popularidade (0,021), com intervalos de
> confiança que **não se sobrepõem**, indicando significância estatística.
> A interpretação prática é que, em **27,8%** das consultas (HR@10), pelo
> menos um item recomendado pelo LTR foi efetivamente acessado pelo usuário
> nas interações subsequentes.

**Discussão qualitativa esperada:**
- Por que valores absolutos parecem baixos (esparsidade — explicar a conta da
  seção 6 deste documento).
- Por que o ganho relativo importa (lift sobre o baseline).
- Trade-offs entre precisão e cobertura/diversidade.
- Limitações da avaliação offline (proxy de relevância = "consumiu depois";
  não captura serendipity nem satisfação).

---

## 8. Glossário rápido

| Termo | Significado |
|---|---|
| **Query** | Uma "consulta" do usuário ao recomendador — no protocolo offline, é a interação de referência. |
| **Gabarito (`future_ids`)** | Conjunto de itens que o usuário **realmente consumiu** depois da referência. É a "verdade fundamental" usada para avaliar. |
| **Top-K** | Lista das K primeiras posições do ranking gerado pelo modelo. |
| **Hit** | Quando um item do Top-K também está no gabarito. |
| **Catálogo** | Conjunto total de itens entre os quais o modelo escolhe. |
| **Esparsidade** | Característica do problema: muitos itens, poucos consumidos por usuário. Faz métricas serem baixas naturalmente. |
| **Baseline** | Modelo simples usado como referência mínima para comparação. |
| **Lift** | Ganho percentual de uma métrica em relação a outro modelo. Ex.: lift de 50% = métrica 1,5× maior. |
| **IC 95%** | Intervalo onde o valor verdadeiro da métrica está com 95% de confiança. |
| **Bootstrap** | Técnica para estimar IC reamostrando os dados com reposição. |
| **Split temporal** | Estratégia em que o conjunto de teste contém as interações mais recentes (preserva ordem temporal — essencial em recsys). |
| **NDCG** | *Normalized Discounted Cumulative Gain* — métrica de ranking sensível à posição. |
| **MRR** | *Mean Reciprocal Rank* — média do recíproco da posição do primeiro acerto. |
| **MAP** | *Mean Average Precision* — média da precisão acumulada nas posições de acerto. |
| **HR** | *Hit Rate* — proporção de queries com pelo menos um acerto. |

---

## Referências para citar no TCC

- **NDCG**: Järvelin & Kekäläinen (2002). "Cumulated Gain-based Evaluation of
  IR Techniques". *ACM TOIS* 20(4).
- **MRR**: Craswell (2009). "Mean Reciprocal Rank". *Encyclopedia of Database
  Systems*. Springer.
- **MAP**: Manning, Raghavan & Schütze (2008). "Introduction to Information
  Retrieval", cap. 8.
- **Avaliação offline em recsys**: Cremonesi, Koren & Turrin (2010).
  "Performance of Recommender Algorithms on Top-N Recommendation Tasks".
  *RecSys '10*.
- **Bootstrap CI**: Efron & Tibshirani (1993). "An Introduction to the
  Bootstrap". Chapman & Hall.
- **Split temporal vs aleatório**: Bergmeir & Benítez (2012). "On the use of
  cross-validation for time series predictor evaluation". *Information
  Sciences* 191.
