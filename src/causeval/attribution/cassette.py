"""Record/replay tool cassettes for deterministic counterfactual rollouts (SPEC 3.6).

Record mode wraps the real tools and stores ``(tool, canonical_args) -> output``. Replay mode
serves recorded outputs on an exact args match, so rollouts from a recorded prefix are
deterministic and free. Interventions produce *novel* tool inputs the recording never saw;
the policy for those:

  * read-only tools marked ``safe_live=True`` are called live (deterministic, no side effects);
  * side-effecting tools are never called live unless ``allow_live_side_effects=True``;
  * anything else falls back to a *simulated* tool (an LLM given the tool schema + recorded
    examples), and the rollout is marked ``fidelity="simulated"``.

:class:`ReplayDivergenceError` is raised when a recorded prefix cannot be reproduced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from causeval.attribution.harness import ToolSpec
from causeval.attribution.trace import Step, canonical_args
from causeval.core.errors import ReplayDivergenceError

Fidelity = str  # "recorded" | "live" | "simulated"
Simulator = Callable[[str, dict[str, Any]], Any]  # (tool, args) -> simulated output


class Cassette:
    """Stores recorded tool outputs keyed by ``tool`` then ``canonical_args``."""

    def __init__(self, recordings: dict[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, dict[str, Any]] = recordings or {}

    def record(self, tool: str, args: dict[str, Any], output: Any) -> None:
        self._store.setdefault(tool, {})[canonical_args(args)] = output

    def has(self, tool: str, args: dict[str, Any]) -> bool:
        return canonical_args(args) in self._store.get(tool, {})

    def get(self, tool: str, args: dict[str, Any]) -> Any:
        return self._store[tool][canonical_args(args)]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {tool: dict(calls) for tool, calls in self._store.items()}


class RecordingBackend:
    """Calls tools live and records every ``(tool, args) -> output`` into a cassette."""

    def __init__(self, specs: dict[str, ToolSpec], cassette: Cassette | None = None) -> None:
        self._specs = specs
        self.cassette = cassette or Cassette()

    def call(self, tool: str, **args: Any) -> Any:
        if tool not in self._specs:
            raise KeyError(f"unknown tool {tool!r}")
        output = self._specs[tool].fn(**args)
        self.cassette.record(tool, args, output)
        return output

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)


class ReplayBackend:
    """Serves recorded outputs; applies the novel-input policy for post-intervention calls."""

    def __init__(
        self,
        specs: dict[str, ToolSpec],
        cassette: Cassette,
        *,
        simulator: Simulator | None = None,
        allow_live_side_effects: bool = False,
    ) -> None:
        self._specs = specs
        self.cassette = cassette
        self._simulator = simulator
        self.allow_live_side_effects = allow_live_side_effects
        self.fidelity: Fidelity = "recorded"

    def call(self, tool: str, **args: Any) -> Any:
        if tool not in self._specs:
            raise KeyError(f"unknown tool {tool!r}")
        if self.cassette.has(tool, args):
            return self.cassette.get(tool, args)

        spec = self._specs[tool]
        if spec.side_effecting and not self.allow_live_side_effects:
            if self._simulator is None:
                raise ReplayDivergenceError(
                    f"novel input to side-effecting tool {tool!r} and no simulator; "
                    "set allow_live_side_effects=True to permit a live call"
                )
            self.fidelity = "simulated"
            return self._simulator(tool, args)
        if spec.safe_live or (spec.side_effecting and self.allow_live_side_effects):
            if self.fidelity == "recorded":
                self.fidelity = "live"
            return spec.fn(**args)
        if self._simulator is not None:
            self.fidelity = "simulated"
            return self._simulator(tool, args)
        raise ReplayDivergenceError(
            f"novel input to tool {tool!r}; mark it safe_live=True or provide a simulator"
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)


def verify_prefix(recorded: list[Step], replayed: list[Step]) -> None:
    """Raise :class:`ReplayDivergenceError` if a replayed prefix diverges from the recording."""
    if len(replayed) < len(recorded):
        raise ReplayDivergenceError(
            f"replay produced {len(replayed)} steps, fewer than the recorded {len(recorded)}"
        )
    for rec, rep in zip(recorded, replayed, strict=False):
        if rec.tool != rep.tool or canonical_args(rec.args) != canonical_args(rep.args):
            raise ReplayDivergenceError(
                f"step {rec.index}: recorded {rec.tool}{rec.args} != replayed {rep.tool}{rep.args}"
            )
        if canonical_args({"o": rec.output}) != canonical_args({"o": rep.output}):
            raise ReplayDivergenceError(
                f"step {rec.index}: output diverged ({rec.output!r} != {rep.output!r})"
            )
