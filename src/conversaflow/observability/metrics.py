"""In-process metrics with Prometheus text exposition.

Counters and histograms are enough for the service's SLIs; exporting in the
Prometheus format keeps the demo dependency-free while remaining scrapeable by
a real Prometheus server.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Iterable


def _labels_text(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items()))
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        key = f"{name}{_labels_text(labels)}"
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = f"{name}{_labels_text(labels)}"
        with self._lock:
            bucket = self._histograms[key]
            bucket.append(float(value))
            if len(bucket) > 5000:
                del bucket[:1000]

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        return self._counters.get(f"{name}{_labels_text(labels)}", 0.0)

    def histogram_values(self, name: str, labels: dict[str, str] | None = None) -> list[float]:
        return list(self._histograms.get(f"{name}{_labels_text(labels)}", []))

    def summary(self, name: str) -> dict[str, float]:
        values = self.histogram_values(name)
        if not values:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        return {
            "count": len(values),
            "p50": round(percentile(values, 0.50), 3),
            "p95": round(percentile(values, 0.95), 3),
            "p99": round(percentile(values, 0.99), 3),
            "mean": round(sum(values) / len(values), 3),
        }

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {key: list(values) for key, values in self._histograms.items()}
        for key, value in sorted(counters.items()):
            lines.append(f"{key} {value:g}")
        for key, values in sorted(histograms.items()):
            if not values:
                continue
            base, label_text = _split_key(key)
            for quantile in (0.5, 0.95, 0.99):
                quantile_label = f'quantile="{quantile:g}"'
                lines.append(
                    f"{_merge_labels(base, label_text, quantile_label)} "
                    f"{percentile(values, quantile):g}"
                )
            lines.append(f"{_merge_labels(base + '_count', label_text)} {len(values)}")
            lines.append(f"{_merge_labels(base + '_sum', label_text)} {sum(values):g}")
        return "\n".join(lines) + ("\n" if lines else "")


def _split_key(key: str) -> tuple[str, str]:
    if "{" not in key:
        return key, ""
    base, _, tail = key.partition("{")
    return base, tail[:-1] if tail.endswith("}") else tail


def _merge_labels(base: str, label_text: str, extra: str | None = None) -> str:
    parts = [part for part in (label_text, extra) if part]
    return f"{base}{{{','.join(parts)}}}" if parts else base
