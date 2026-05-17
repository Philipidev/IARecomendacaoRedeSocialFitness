"""
Servidor FastAPI do simulador FitConnect.

Sobe em porta separada (8001) para não conflitar com o dashboard do pipeline
(`web/app.py`, porta 8000). Expõe três endpoints JSON e serve o frontend
estático em HTML/CSS/JS puro.

Uso:
    python -m simulador.api
    # ou
    uvicorn simulador.api:app --reload --port 8001
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulador import service

SIM_DIR = Path(__file__).resolve().parent
STATIC_DIR = SIM_DIR / "static"
TEMPLATES_DIR = SIM_DIR / "templates"

app = FastAPI(title="FitConnect — Simulador de Recomendações")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/api/models")
async def api_models():
    return {"models": service.discover_models()}


@app.get("/api/tags")
async def api_tags(model_dir: str):
    if not model_dir:
        raise HTTPException(status_code=400, detail="model_dir é obrigatório")
    try:
        return service.list_tags(model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - guarda de defesa
        raise HTTPException(status_code=500, detail=str(exc))


class RecommendRequest(BaseModel):
    model_dir: str
    tags: list[str] = Field(default_factory=list)
    top_k: int = 20
    user_id: int | None = None
    timestamp: int | None = None
    excluir_tags_exatas: bool = False


@app.post("/api/recommend")
async def api_recommend(body: RecommendRequest):
    try:
        payload = service.recommend(
            model_dir=body.model_dir,
            tags=body.tags,
            top_k=body.top_k,
            user_id=body.user_id,
            timestamp=body.timestamp,
            excluir_tags_exatas=body.excluir_tags_exatas,
        )
        return JSONResponse(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - guarda de defesa
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("simulador.api:app", host="0.0.0.0", port=8001, reload=False)
