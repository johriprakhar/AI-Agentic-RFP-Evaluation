"""
Explicit tool registry and execution trace.

The evaluation pipeline is an orchestrated sequence of capabilities, most of
them deterministic Python and exactly one of them a language model. Registering
those capabilities as named tools and routing every call through the registry
buys three things that matter for an auditable procurement decision:

* a single declaration of which steps are deterministic and which are model
  judgement, visible in the UI and stored with the run;
* a timed, ordered execution trace per supplier, so a reviewer can see what ran,
  in what order, how long it took and what it produced;
* one uniform place where a failing step is captured rather than crashing the
  run, so one unreadable PDF cannot take down an otherwise valid evaluation.

The registry intentionally does not let the model choose which tool to call.
In a scoring pipeline the sequence is fixed and must be reproducible; handing
step selection to a model would trade determinism for nothing. The model is
confined to the single judgement step where its judgement is the point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: Tool kinds, used to render the separation of concerns in the UI.
DETERMINISTIC = "deterministic"
MODEL = "model"


@dataclass(frozen=True)
class Tool:
    """One named capability available to the orchestrator."""

    name: str
    kind: str
    description: str
    run: Callable[..., Any]

    @property
    def is_model(self) -> bool:
        return self.kind == MODEL


@dataclass
class ToolInvocation:
    """A single recorded tool call."""

    name: str
    kind: str
    ok: bool
    duration_ms: float
    summary: str = ""
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "kind": self.kind,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class ToolTrace:
    """An ordered, timed log of the tool calls made for one supplier or run."""

    label: str = ""
    invocations: List[ToolInvocation] = field(default_factory=list)

    def add(self, invocation: ToolInvocation) -> None:
        self.invocations.append(invocation)

    @property
    def total_ms(self) -> float:
        return sum(item.duration_ms for item in self.invocations)

    @property
    def failed(self) -> List[ToolInvocation]:
        return [item for item in self.invocations if not item.ok]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "total_ms": round(self.total_ms, 2),
            "model_calls": sum(1 for item in self.invocations if item.kind == MODEL),
            "deterministic_calls": sum(1 for item in self.invocations if item.kind == DETERMINISTIC),
            "invocations": [item.as_dict() for item in self.invocations],
        }


class ToolRegistry:
    """A small, explicit registry of the pipeline's capabilities."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        run: Callable[..., Any],
        kind: str = DETERMINISTIC,
        description: str = "",
    ) -> Tool:
        tool = Tool(name=name, kind=kind, description=description, run=run)
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Registered: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> List[Dict[str, str]]:
        """Manifest of every registered tool, stored with each run."""
        return [
            {"tool": tool.name, "kind": tool.kind, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def invoke(
        self,
        name: str,
        *args,
        trace: Optional[ToolTrace] = None,
        summarize: Optional[Callable[[Any], str]] = None,
        **kwargs,
    ) -> Any:
        """
        Run a registered tool, timing it and recording the outcome on ``trace``.

        Exceptions are recorded and re-raised: the trace keeps the evidence, the
        caller keeps control over whether a failure is fatal for that supplier.
        """
        tool = self.get(name)
        started = time.perf_counter()
        try:
            result = tool.run(*args, **kwargs)
        except Exception as error:
            if trace is not None:
                trace.add(ToolInvocation(
                    name=name,
                    kind=tool.kind,
                    ok=False,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    error=f"{type(error).__name__}: {error}",
                ))
            raise

        if trace is not None:
            summary = ""
            if summarize is not None:
                try:
                    summary = summarize(result)
                except Exception:
                    summary = ""
            trace.add(ToolInvocation(
                name=name,
                kind=tool.kind,
                ok=True,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                summary=summary,
            ))
        return result
