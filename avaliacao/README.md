# Avaliação automática (antes/depois)

Este módulo compara o desempenho do ranking com e sem o sinal de popularidade.

## Script

```bash
# Modo real (usa artefatos em treinamento/dados e treinamento/modelo)
python avaliacao/avaliar_popularidade.py --k 10 --peso-depois 0.10

# Modo demo (sem dependência de dados locais)
python avaliacao/avaliar_popularidade.py --demo --k 10 --peso-depois 0.10
```

## Saída

Gera `avaliacao/metricas_antes_depois.json` com:
- `antes`: métricas com `peso_popularidade=0.0`
- `depois`: métricas com `peso_popularidade=<valor configurado>`
- `delta`: variação (`depois - antes`)
