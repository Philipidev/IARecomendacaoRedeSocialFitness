from __future__ import annotations

from dataclasses import dataclass, field


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _percent(current: int, total: int) -> int:
    if total <= 0:
        return 100
    return int(round((_clamp(current, 0, total) / total) * 100))


def _bucket(percent: int, every_percent: int) -> int:
    if every_percent <= 0:
        return percent
    return min(100, (percent // every_percent) * every_percent)


@dataclass
class IterationProgress:
    total: int
    label: str
    every_percent: int = 5
    current: int = 0
    _last_bucket: int = field(default=-1, init=False)

    def log(
        self,
        current: int | None = None,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        if current is not None:
            self.current = current

        total = max(int(self.total), 0)
        current_value = _clamp(int(self.current), 0, total if total > 0 else int(self.current))
        percent = _percent(current_value, total)
        bucket = _bucket(percent, self.every_percent)

        if not force and bucket <= self._last_bucket:
            return

        suffix = f" - {detail}" if detail else ""
        print(f"[{self.label}] {current_value}/{total} ({percent} %){suffix}")
        self._last_bucket = bucket

    def start(self, detail: str | None = None) -> None:
        self.log(current=0, detail=detail, force=True)

    def advance(self, step: int = 1, detail: str | None = None) -> None:
        self.log(current=self.current + step, detail=detail)

    def finish(self, detail: str | None = None) -> None:
        target = max(int(self.total), int(self.current))
        self.log(current=target, detail=detail, force=True)


@dataclass
class StageProgress:
    total_stages: int
    label: str
    current_stage: int = 0

    def step(self, detail: str) -> None:
        self.current_stage += 1
        total = max(int(self.total_stages), 1)
        current = _clamp(int(self.current_stage), 1, total)
        percent = _percent(current, total)
        print(f"[{self.label}] {current}/{total} ({percent} %) - {detail}")
