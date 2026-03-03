---
name: Exportar Métricas TXT
overview: Estender `preparacao_dados.py` para exportar um arquivo `.txt` por métrica relevante dos Parquets de output, onde cada linha contém um valor único daquela dimensão. Os arquivos ficam em `treinamento/dados/`.
todos:
  - id: add-cooccurrence-load
    content: Adicionar 'tag_cooccurrence' e 'user_social_graph' à lista de parquets carregados em carregar_parquets()
    status: completed
  - id: add-salvar-metricas
    content: Implementar função salvar_metricas_txt e chamá-la no main()
    status: completed
  - id: update-readme
    content: Atualizar treinamento/README.md para listar os novos .txt em treinamento/dados/
    status: completed
isProject: false
---

# Exportar Métricas como Arquivos .txt

## Arquivos gerados em `treinamento/dados/`

Cada `.txt` terá **um valor por linha**, ordenado alfabética ou numericamente.

- `tag_lista.txt` — já existe; gerado de `tags_fitness.parquet` → coluna `tag_name`
- `event_type_lista.txt` — valores únicos de `event_type` de `interactions_fitness.parquet` (like, create, reply)
- `language_lista.txt` — valores únicos de `language` de `messages_fitness.parquet` (excluindo nulos)
- `message_type_lista.txt` — valores únicos de `message_type` de `messages_fitness.parquet` (post, comment)
- `user_id_lista.txt` — todos os `user_id` únicos de `interactions_fitness.parquet`
- `tag_cooccurrence_pares_lista.txt` — pares `tag_a|tag_b|cooccurrences` de `tag_cooccurrence.parquet`, ordenados por `cooccurrences` desc (formato diferente: 3 campos por linha separados por `|`)

## Arquivo alterado

Somente `[treinamento/preparacao_dados.py](treinamento/preparacao_dados.py)` — adicionar uma função `salvar_metricas_txt` chamada no `main()`, após o bloco de `salvar_tag_lista`.

## Lógica da nova função

```python
def salvar_metricas_txt(dados: dict[str, pd.DataFrame], dados_dir: Path) -> None:
    # event_type_lista.txt
    if "interactions_fitness" in dados:
        vals = sorted(dados["interactions_fitness"]["event_type"].dropna().unique())
        (dados_dir / "event_type_lista.txt").write_text("\n".join(vals), encoding="utf-8")

    # language_lista.txt
    if "messages_fitness" in dados:
        vals = sorted(dados["messages_fitness"]["language"].dropna().unique())
        (dados_dir / "language_lista.txt").write_text("\n".join(str(v) for v in vals), encoding="utf-8")

    # message_type_lista.txt
    if "messages_fitness" in dados:
        vals = sorted(dados["messages_fitness"]["message_type"].dropna().unique())
        (dados_dir / "message_type_lista.txt").write_text("\n".join(vals), encoding="utf-8")

    # user_id_lista.txt
    if "interactions_fitness" in dados:
        vals = sorted(dados["interactions_fitness"]["user_id"].dropna().unique())
        (dados_dir / "user_id_lista.txt").write_text("\n".join(str(v) for v in vals), encoding="utf-8")

    # tag_cooccurrence_pares_lista.txt
    if "tag_cooccurrence" in dados:
        df = dados["tag_cooccurrence"].sort_values("cooccurrences", ascending=False)
        linhas = [f"{r.tag_a}|{r.tag_b}|{r.cooccurrences}" for r in df.itertuples()]
        (dados_dir / "tag_cooccurrence_pares_lista.txt").write_text("\n".join(linhas), encoding="utf-8")
```

## Observações

- `tag_cooccurrence` já é carregado em `carregar_parquets()` — basta adicionar `"tag_cooccurrence"` à lista `nomes` nessa função (atualmente não está lá).
- `user_social_graph.parquet` **não** gera .txt próprio: os valores (`user_id`, `friend_id`) já ficam cobertos pelo `user_id_lista.txt`.
- Os arquivos são sobrescritos a cada execução de `preparacao_dados.py`, assim como os demais artefatos.

