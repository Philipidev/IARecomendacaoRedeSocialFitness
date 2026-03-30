"""
Executor de subprocessos com streaming de stdout/stderr.

Garante execucao unica (lock de concorrencia) e distribui linhas
de log em tempo real para assinantes SSE.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from web.progress_parser import parse_progress_line

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / ".execution_history.json"
LOGS_DIR = ROOT / ".execution_logs"


def _load_history() -> list[dict[str, Any]]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(history: list[dict[str, Any]]) -> None:
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class ProcessExecutor:
    """Executor singleton com lock de concorrencia e streaming SSE."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._current_process: subprocess.Popen | None = None
        self._current_task: str | None = None
        self._current_id: str | None = None
        self._log_lines: list[str] = []
        self._subscribers: list[asyncio.Queue] = []
        self._status: str = "idle"  # idle | running | completed | error | cancelled
        self._last_progress: dict[str, Any] | None = None

    def is_running(self) -> bool:
        return self._status == "running"

    def get_status(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "task": self._current_task,
            "execution_id": self._current_id,
            "log_count": len(self._log_lines),
            "last_progress": self._last_progress,
        }

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        """Gera eventos SSE para um cliente. Envia logs anteriores e depois fica em tempo real."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)

        # Enviar log lines ja acumulados
        for line in list(self._log_lines):
            yield {"type": "log", "data": line}
        if self._last_progress:
            yield {"type": "progress", "data": self._last_progress}
        yield {"type": "status", "data": self._status}

        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "status" and event.get("data") in ("completed", "error", "cancelled"):
                    break
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def _broadcast(self, event: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def cancel(self) -> bool:
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()
            self._status = "cancelled"
            self._broadcast({"type": "status", "data": "cancelled"})
            return True
        return False

    async def execute(
        self,
        label: str,
        steps: list[tuple[str, Path, list[str]]],
        *,
        on_complete: Any | None = None,
    ) -> dict[str, Any]:
        """Executa uma sequencia de subprocessos com streaming.

        Args:
            label: Nome da acao (ex: "Rodar extracao")
            steps: Lista de (step_label, script_path, args)
            on_complete: Callable(state) opcional, chamado ao final

        Returns:
            Registro de execucao
        """
        if self._lock.locked():
            raise RuntimeError("Ja existe uma execucao em andamento.")

        async with self._lock:
            execution_id = str(uuid.uuid4())[:8]
            self._current_id = execution_id
            self._current_task = label
            self._log_lines = []
            self._last_progress = None
            self._status = "running"

            LOGS_DIR.mkdir(exist_ok=True)
            log_file = LOGS_DIR / f"{execution_id}.log"

            self._broadcast({"type": "status", "data": "running"})
            self._broadcast({"type": "task", "data": {"label": label, "id": execution_id}})

            started_at = datetime.now(timezone.utc).isoformat()
            exit_code = 0
            final_status = "completed"

            total_steps = len(steps)
            for step_idx, (step_label, script, args) in enumerate(steps, 1):
                if self._status == "cancelled":
                    final_status = "cancelled"
                    break

                stage_event = {
                    "type": "stage",
                    "data": {
                        "label": step_label,
                        "current": step_idx,
                        "total": total_steps,
                        "percent": int(round(step_idx / total_steps * 100)),
                        "script": str(script.name),
                    },
                }
                self._broadcast(stage_event)
                self._append_log(f"\n=== [{step_idx}/{total_steps}] {step_label} ===")
                self._append_log(f"[Execucao] python {script.name} {' '.join(args)}\n")

                try:
                    exit_code = await self._run_single(script, args, log_file)
                    if exit_code != 0:
                        self._append_log(f"\n[Erro] {script.name} falhou com codigo {exit_code}.")
                        final_status = "error"
                        break
                except Exception as exc:
                    self._append_log(f"\n[Erro] Excecao: {exc}")
                    final_status = "error"
                    exit_code = 1
                    break

            if self._status == "cancelled":
                final_status = "cancelled"

            self._status = final_status
            finished_at = datetime.now(timezone.utc).isoformat()

            self._broadcast({"type": "status", "data": final_status})
            self._broadcast({"type": "state_updated", "data": {}})

            record = {
                "id": execution_id,
                "action": label,
                "started_at": started_at,
                "finished_at": finished_at,
                "status": final_status,
                "exit_code": exit_code,
                "log_file": str(log_file.relative_to(ROOT)),
                "steps": total_steps,
            }
            self._save_to_history(record)

            if on_complete and final_status == "completed":
                try:
                    on_complete()
                except Exception:
                    pass

            self._current_process = None
            return record

    async def _run_single(self, script: Path, args: list[str], log_file: Path) -> int:
        """Executa um unico subprocess com streaming."""
        cmd = [sys.executable, "-u", str(script), *args]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = await asyncio.to_thread(
            lambda: subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        )
        self._current_process = process

        def read_output():
            assert process.stdout is not None
            for line in process.stdout:
                stripped = line.rstrip("\n")
                self._append_log(stripped)

                progress = parse_progress_line(stripped)
                if progress:
                    self._last_progress = progress
                    self._broadcast({"type": "progress", "data": progress})
                else:
                    self._broadcast({"type": "log", "data": stripped})

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line)

            process.wait()

        await asyncio.to_thread(read_output)
        return process.returncode or 0

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)

    def _save_to_history(self, record: dict[str, Any]) -> None:
        history = _load_history()
        history.insert(0, record)
        # Manter no maximo 100 registros
        history = history[:100]
        _save_history(history)


# Singleton
executor = ProcessExecutor()
