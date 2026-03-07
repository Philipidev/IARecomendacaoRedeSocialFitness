# Avaliação do Sistema de Recomendação

Este diretório reúne dois fluxos complementares:

- **avaliação automática** do impacto do sinal de popularidade no ranking;
- **avaliação manual reproduzível** para validação qualitativa por banca/orientador.

## Avaliação automática (antes/depois)

Compara o desempenho do ranking com e sem o sinal de popularidade.

### Script

```bash
# Modo real (usa artefatos em treinamento/dados e treinamento/modelo)
python avaliacao/avaliar_popularidade.py --k 10 --peso-depois 0.10

# Modo demo (sem dependência de dados locais)
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

Define um fluxo manual e reproduzível para validar qualitativamente as recomendações.

### Arquivos

- `avaliacao/casos_manuais.yaml`: cenários de teste com entradas e critérios esperados.
- `avaliacao/avaliacao_manual.py`: executor dos cenários; gera top-K e compara resultado observado vs esperado.
- `avaliacao/resultados/avaliacao_manual.md`: relatório consolidado gerado automaticamente.

### Como executar

```bash
python avaliacao/avaliacao_manual.py
```

Opcionalmente, informe caminhos customizados:

```bash
python avaliacao/avaliacao_manual.py --casos avaliacao/casos_manuais.yaml --saida avaliacao/resultados/avaliacao_manual.md
```

### Formato dos cenários (`casos_manuais.yaml`)

Cada caso contém:

1. **Entrada**
- `tags`: lista de tags de entrada para recomendação
- `timestamp`: instante de referência em milissegundos
- `top_k`: quantidade de resultados esperada
- `contexto_usuario`: perfil, objetivo e restrições para apoiar a análise qualitativa

2. **Critérios de saída esperada**
- `aderencia_tematica`
- `esperado`: descrição textual
- `proporcao_minima`: valor entre 0 e 1
- `recencia`
- `esperado`: descrição textual
- `max_delta_dias`: janela temporal
- `proporcao_minima`: valor entre 0 e 1
- `variedade`
- `esperado`: descrição textual
- `minimo_conjuntos_unicos`
- `ausencia_itens_irrelevantes`
- `esperado`: descrição textual
- `tags_bloqueadas`: lista de tags que não devem aparecer
- `proporcao_minima`: valor entre 0 e 1

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

- tabela com o top-K retornado pelo modelo
- tabela de critérios qualitativos com esperado, observado e status (`✅`/`❌`)
- conclusão geral com quantidade de cenários aprovados integralmente

Se os artefatos de treino/modelo não estiverem disponíveis, o script registra erro
no caso e ainda gera o relatório, permitindo repetir o processo após preparar o ambiente.
