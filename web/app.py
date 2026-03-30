"""
Servidor FastAPI — interface web do orquestrador de pipeline fitness.

Uso:
    python -m web.app
    # ou
    uvicorn web.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Garantir que o root do projeto esteja no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_service as svc
from web.executor import executor, _load_history

app = FastAPI(title="Pipeline Fitness — Dashboard")

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state() -> dict[str, Any]:
    """Carrega e atualiza o estado do pipeline."""
    return svc.update_and_save(svc.load_state())


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    """Wrapper para TemplateResponse compativel com Starlette >=1.0."""
    return templates.TemplateResponse(request, template, context)


# ---------------------------------------------------------------------------
# Pagina principal
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    state = _fresh_state()
    details = svc.service_get_state_details(state)
    return _render(request, "dashboard.html", {
        "state": state,
        "details": details,
        "exec_status": executor.get_status(),
        "history": _load_history()[:20],
    })


# ---------------------------------------------------------------------------
# Partials HTMX
# ---------------------------------------------------------------------------

@app.get("/partials/status-header", response_class=HTMLResponse)
async def partial_status_header(request: Request):
    state = _fresh_state()
    details = svc.service_get_state_details(state)
    return _render(request, "partials/status_header.html", {
        "details": details,
        "exec_status": executor.get_status(),
    })


@app.get("/partials/status-cards", response_class=HTMLResponse)
async def partial_status_cards(request: Request):
    state = _fresh_state()
    details = svc.service_get_state_details(state)
    return _render(request, "partials/status_cards.html", {
        "details": details,
    })


@app.get("/partials/actions-panel", response_class=HTMLResponse)
async def partial_actions_panel(request: Request):
    state = _fresh_state()
    details = svc.service_get_state_details(state)
    targets = svc.service_list_model_targets(state)
    family = str((state.get("selected_model_target") or {}).get("family", "baseline_hibrido"))
    eval_modes = svc.service_get_eval_modes(family)
    return _render(request, "partials/actions_panel.html", {
        "details": details,
        "datasets": details["datasets"],
        "targets": targets,
        "eval_modes": eval_modes,
        "download_options": svc.DOWNLOAD_OPTIONS,
        "exec_status": executor.get_status(),
    })


@app.get("/partials/execution-panel", response_class=HTMLResponse)
async def partial_execution_panel(request: Request):
    return _render(request, "partials/execution_panel.html", {
        "exec_status": executor.get_status(),
    })


@app.get("/partials/history-panel", response_class=HTMLResponse)
async def partial_history_panel(request: Request):
    return _render(request, "partials/history_panel.html", {
        "history": _load_history()[:20],
    })


# ---------------------------------------------------------------------------
# API JSON — Estado
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def api_state():
    state = _fresh_state()
    return svc.service_get_state_details(state)


@app.get("/api/datasets")
async def api_datasets():
    state = _fresh_state()
    return state["workspace"]["datasets"]


@app.get("/api/models")
async def api_models():
    state = _fresh_state()
    return svc.service_list_model_targets(state)


@app.get("/api/download-options")
async def api_download_options():
    return svc.DOWNLOAD_OPTIONS


@app.get("/api/eval-modes/{family}")
async def api_eval_modes(family: str):
    return svc.service_get_eval_modes(family)


@app.get("/api/execution/status")
async def api_execution_status():
    return executor.get_status()


@app.get("/api/history")
async def api_history():
    return _load_history()


# ---------------------------------------------------------------------------
# API — Selecao (acoes rapidas, sem subprocess)
# ---------------------------------------------------------------------------

class SelectDatasetRequest(BaseModel):
    index: int

@app.post("/api/select-dataset")
async def api_select_dataset(body: SelectDatasetRequest):
    try:
        state = _fresh_state()
        state = svc.service_select_dataset(state, body.index)
        return {"ok": True, "dataset": svc.selected_dataset_label(state)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DownloadDatasetRequest(BaseModel):
    scale_factor: str

class SelectModelRequest(BaseModel):
    index: int

@app.post("/api/select-model")
async def api_select_model(body: SelectModelRequest):
    try:
        state = _fresh_state()
        state = svc.service_select_model_target(state, body.index)
        return {"ok": True, "model": svc.selected_model_target_label(state)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SelectBenchmarkRequest(BaseModel):
    scope: str = "all"
    model_ids: list[str] | None = None

@app.post("/api/select-benchmark")
async def api_select_benchmark(body: SelectBenchmarkRequest):
    try:
        state = _fresh_state()
        state = svc.service_select_benchmark(state, body.scope, body.model_ids)
        return {"ok": True, "benchmark": svc.benchmark_target_label(state)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# API — Execucao (subprocessos com streaming)
# ---------------------------------------------------------------------------

def _check_not_running():
    if executor.is_running():
        raise HTTPException(status_code=409, detail="Ja existe uma execucao em andamento.")


@app.post("/api/run/extraction")
async def api_run_extraction():
    _check_not_running()
    state = _fresh_state()
    try:
        script, args = svc.build_extraction_args(state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    steps = [("Extracao", script, args)]

    def on_complete():
        s = _fresh_state()
        svc.register_run(s, "extraction", {
            "dataset_key": (s.get("selected_dataset") or {}).get("dataset_key"),
        })
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Rodar extracao", steps, on_complete=on_complete))
    return {"ok": True, "message": "Extracao iniciada."}


@app.post("/api/download-dataset")
async def api_download_dataset(body: DownloadDatasetRequest):
    _check_not_running()
    opt = svc.find_download_option(body.scale_factor)
    if not opt:
        raise HTTPException(status_code=400, detail=f"Scale factor invalido: {body.scale_factor}")

    steps = [("Download dataset", svc.DOWNLOAD_SCRIPT, ["--scale-factor", body.scale_factor])]

    def on_complete():
        s = _fresh_state()
        dataset_path = svc.DATASET_DIR / opt["filename"]
        if dataset_path.exists():
            svc.update_selected_dataset(s, dataset_path, scale_factor=opt["scale_factor"], source="download")
            svc.register_run(s, "download", {
                "scale_factor": opt["scale_factor"],
            })
            svc.update_and_save(s)

    asyncio.create_task(executor.execute("Baixar dataset", steps, on_complete=on_complete))
    return {"ok": True, "message": f"Download de {body.scale_factor} iniciado."}


class RunTrainingRequest(BaseModel):
    split_config: dict[str, Any] | None = None

@app.post("/api/run/training")
async def api_run_training(body: RunTrainingRequest):
    _check_not_running()
    state = _fresh_state()
    try:
        steps = svc.build_training_args(state, body.split_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def on_complete():
        s = _fresh_state()
        target = svc.get_selected_model_target(s)
        svc.register_run(s, "training", {"model_target": dict(target)})
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Rodar treinamento", steps, on_complete=on_complete))
    return {"ok": True, "message": "Treinamento iniciado."}


class RunEvaluationRequest(BaseModel):
    modes: list[str]

@app.post("/api/run/evaluation")
async def api_run_evaluation(body: RunEvaluationRequest):
    _check_not_running()
    state = _fresh_state()
    try:
        steps = svc.build_evaluation_args(state, body.modes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def on_complete():
        s = _fresh_state()
        svc.register_run(s, "evaluation", {"modes": body.modes})
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Rodar avaliacao", steps, on_complete=on_complete))
    return {"ok": True, "message": "Avaliacao iniciada."}


class RunBenchmarkRequest(BaseModel):
    scope: str | None = None
    model_ids: list[str] | None = None

@app.post("/api/run/benchmark")
async def api_run_benchmark(body: RunBenchmarkRequest):
    _check_not_running()
    state = _fresh_state()

    if body.scope:
        state = svc.service_select_benchmark(state, body.scope, body.model_ids)

    try:
        script, args = svc.build_benchmark_args(state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    steps = [("Benchmark TCC", script, args)]

    def on_complete():
        s = _fresh_state()
        svc.register_run(s, "benchmark_tcc", {
            "selection": dict(s.get("selected_benchmark", svc.default_benchmark_target())),
        })
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Benchmark TCC", steps, on_complete=on_complete))
    return {"ok": True, "message": "Benchmark iniciado."}


class RunFullPipelineRequest(BaseModel):
    split_config: dict[str, Any] | None = None
    modes: list[str] | None = None

@app.post("/api/run/full-pipeline")
async def api_run_full_pipeline(body: RunFullPipelineRequest):
    _check_not_running()
    state = _fresh_state()

    try:
        ext_script, ext_args = svc.build_extraction_args(state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        train_steps = svc.build_training_args(state, body.split_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    modes = body.modes or ["offline", "manual"]
    try:
        eval_steps = svc.build_evaluation_args(state, modes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    all_steps = [("Extracao", ext_script, ext_args)] + train_steps + eval_steps

    def on_complete():
        s = _fresh_state()
        svc.register_run(s, "full_pipeline", {"modes": modes})
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Pipeline completo", all_steps, on_complete=on_complete))
    return {"ok": True, "message": "Pipeline completo iniciado."}


class RunTrainingEvalRequest(BaseModel):
    split_config: dict[str, Any] | None = None
    modes: list[str] | None = None

@app.post("/api/run/training-evaluation")
async def api_run_training_evaluation(body: RunTrainingEvalRequest):
    _check_not_running()
    state = _fresh_state()

    try:
        train_steps = svc.build_training_args(state, body.split_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    modes = body.modes or ["offline", "manual"]
    try:
        eval_steps = svc.build_evaluation_args(state, modes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    all_steps = train_steps + eval_steps

    def on_complete():
        s = _fresh_state()
        svc.register_run(s, "training_evaluation", {"modes": modes})
        svc.update_and_save(s)

    asyncio.create_task(executor.execute("Treinamento + Avaliacao", all_steps, on_complete=on_complete))
    return {"ok": True, "message": "Treinamento + Avaliacao iniciado."}


@app.post("/api/cancel")
async def api_cancel():
    if executor.cancel():
        return {"ok": True, "message": "Execucao cancelada."}
    raise HTTPException(status_code=400, detail="Nenhuma execucao em andamento.")


# ---------------------------------------------------------------------------
# SSE — Streaming de execucao
# ---------------------------------------------------------------------------

@app.get("/api/run/stream")
async def api_run_stream():
    async def event_generator():
        async for event in executor.subscribe():
            event_type = event.get("type", "message")
            data = event.get("data", "")
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
