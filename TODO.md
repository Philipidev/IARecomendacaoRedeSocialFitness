# TODO

- [ ] Implementar o `LTR v2` personalizado por `user_id`.
  Status atual: o projeto possui `LTR v1` padrão/anônimo; a personalização do LTR ainda não foi concluída.
  Pendências principais:
  - usar `query_user_id` de forma efetiva na montagem do dataset LTR
  - incluir features usuário-item no treino do ranker, como `user_affinity_score`
  - alinhar treino, inferência e benchmark multi-modelo para comparar `LTR v1` vs `LTR v2`
