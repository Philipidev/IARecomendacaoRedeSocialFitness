# Avaliação do Sistema de Recomendação

Este diretório reúne quatro frentes complementares:

- avaliação offline do recomendador no split de teste
- avaliação automática do impacto do sinal de popularidade
- avaliação manual reproduzível para análise qualitativa
- otimização de pesos do score híbrido

## Scripts disponíveis

- `avaliacao/avaliar_modelo.py`: mede métricas de ranking e de negócio no split de teste
- `avaliacao/avaliar_popularidade.py`: compara o ranking antes/depois do peso de popularidade
- `avaliacao/avaliacao_manual.py`: executa cenários qualitativos reproduzíveis
- `avaliacao/otimizar_pesos.py`: busca pesos melhores para os quatro sinais base e grava `treinamento/modelo/pesos_otimos.json`

## Avaliação offline do recomendador

Executa a avaliação do modelo treinado em `treinamento/modelo/` usando os dados
de teste em `treinamento/dados/splits/`.

### Comando

```bash
python avaliacao/avaliar_modelo.py --k 5 10 20
```

O script grava resultados em `avaliacao/resultados/` nos formatos JSON, CSV e
Markdown.

### Protocolo de avaliação

1. Carrega o split de teste (`test_interactions.parquet`).
2. Para cada usuário, ordena as interações por tempo.
3. Cada interação vira um item de referência.
4. O ground truth é o conjunto de interações futuras do mesmo usuário no teste.
5. Gera recomendações a partir das tags e do timestamp do item de referência.
6. Compara o Top-K recomendado com os itens futuros reais para calcular métricas.

### Métricas de ranking em K

- `Precision@K`: fração dos K recomendados que são relevantes
- `Recall@K`: fração dos relevantes recuperados no Top-K
- `HitRate@K`: 1 se houver ao menos um item relevante no Top-K; senão 0
- `MAP@K`: média da precisão acumulada nas posições com hit, normalizada por `min(|relevantes|, K)`
- `NDCG@K`: qualidade do ranqueamento considerando posição

### Métricas de negócio/TCC

- cobertura de catálogo
- diversidade intra-lista por tags
- novidade baseada em popularidade inversa
- recência média recomendada em dias

### Arquivos gerados

- `avaliacao/resultados/metricas_resumo.json`
- `avaliacao/resultados/metricas_ranking_por_k.csv`
- `avaliacao/resultados/queries_avaliadas.csv`
- `avaliacao/resultados/resumo_avaliacao.md`

## Avaliação automática de popularidade

Compara o desempenho do ranking com e sem o sinal de popularidade.

### Comandos

```bash
# Modo real
python avaliacao/avaliar_popularidade.py --k 10 --peso-depois 0.10

# Modo demo
python avaliacao/avaliar_popularidade.py --demo --k 10 --peso-depois 0.10
```

### Saída

Gera `avaliacao/metricas_antes_depois.json` com:

- `modo`: `real` ou `demo`
- `config`: configuração usada na execução
- `antes`: métricas com `peso_popularidade=0.0`
- `depois`: métricas com `peso_popularidade=<valor configurado>`
- `delta`: variação (`depois - antes`) para as métricas de ranking

## Avaliação manual reproduzível

Define um fluxo manual e reproduzível para validar qualitativamente as
recomendações.

### Arquivos

- `avaliacao/casos_manuais.yaml`: cenários com entradas e critérios esperados
- `avaliacao/avaliacao_manual.py`: executor dos cenários
- `avaliacao/resultados/avaliacao_manual.md`: relatório consolidado gerado automaticamente

### Como executar

```bash
python avaliacao/avaliacao_manual.py
```

Opcionalmente:

```bash
python avaliacao/avaliacao_manual.py --casos avaliacao/casos_manuais.yaml --saida avaliacao/resultados/avaliacao_manual.md
```

### Estrutura dos cenários

Cada caso em `casos_manuais.yaml` define:

- entrada com `tags`, `timestamp`, `top_k` e `contexto_usuario`
- critérios de saída esperada para aderência temática, recência, variedade e ausência de itens irrelevantes

### Exemplo mínimo

```yaml
casos:
  - id: caso_exemplo
    titulo: "Exemplo"
    descricao: "Descrição do cenário"
    entrada:
      tags: ["Born_to_Run"]
      timestamp: 1320000000000
      top_k: 5
      contexto_usuario:
        perfil: "iniciante"
        objetivo: "corrida"
        restricoes: []
    criterios_saida:
      aderencia_tematica:
        esperado: "Maioria dos itens no tema"
        proporcao_minima: 0.8
      recencia:
        esperado: "Resultados próximos no tempo"
        max_delta_dias: 365
        proporcao_minima: 0.6
      variedade:
        esperado: "Diversidade de combinações"
        minimo_conjuntos_unicos: 3
      ausencia_itens_irrelevantes:
        esperado: "Sem itens fora de escopo"
        tags_bloqueadas: ["Politics"]
        proporcao_minima: 1.0
```

### Interpretação do relatório

Para cada caso, o markdown final traz:

- tabela com o Top-K retornado pelo modelo
- tabela de critérios qualitativos com esperado, observado e status
- conclusão geral com a quantidade de cenários aprovados integralmente

Se os artefatos de treino/modelo não estiverem disponíveis, o script registra o
erro no caso e ainda gera o relatório.

## Otimização de pesos

Executa grid search e, opcionalmente, random search para ajustar os pesos dos
quatro sinais base do score híbrido.

### Comando

```bash
python avaliacao/otimizar_pesos.py --grid-step 0.1 --top-k 10 --max-queries 300
```

### Saídas

- `avaliacao/resultados/pesos_experimentos.csv`
- `treinamento/modelo/pesos_otimos.json`
