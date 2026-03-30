"""
Parser de linhas de progresso do pipeline.

Reconhece o formato emitido por progress_utils.py:
  [Label] N/M (X %) - detalhe
"""

from __future__ import annotations

import re

PROGRESS_RE = re.compile(
    r"\[(?P<label>[^\]]+)\]\s*"
    r"(?P<current>\d+)/(?P<total>\d+)\s*"
    r"\((?P<percent>\d+)\s*%\)"
    r"(?:\s*-\s*(?P<detail>.+))?"
)


def parse_progress_line(line: str) -> dict | None:
    """Tenta extrair informacao de progresso de uma linha de log.

    Retorna dict com label, current, total, percent, detail ou None se
    a linha nao corresponde ao formato esperado.
    """
    m = PROGRESS_RE.search(line)
    if not m:
        return None
    return {
        "label": m.group("label"),
        "current": int(m.group("current")),
        "total": int(m.group("total")),
        "percent": int(m.group("percent")),
        "detail": (m.group("detail") or "").strip() or None,
    }
