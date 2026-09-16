"""DSPy retrieval module backed by dynavec.

    import dspy
    from dynavec.integrations.dspy import DynavecRM

    rm = DynavecRM(dynavec_client, namespace="kb", k=4)
    dspy.configure(rm=rm)

    retriever = dspy.Retrieve(k=4)
    passages = retriever("what is retrieval-augmented generation?").passages
"""

from __future__ import annotations

from typing import Any

from ..client import Dynavec
from ..exceptions import MissingDependencyError

try:
    import dspy
    from dspy.dsp.utils import dotdict
except ImportError as exc:  # pragma: no cover - import guard
    raise MissingDependencyError("DynavecRM", "dspy", "dspy") from exc


class DynavecRM(dspy.Retrieve):
    """DSPy retrieval module backed by a :class:`Dynavec` client."""

    def __init__(
        self,
        client: Dynavec,
        namespace: str = "default",
        k: int = 3,
    ) -> None:
        self._client = client
        self._namespace = namespace
        super().__init__(k=k)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs: Any,
    ) -> list[dotdict]:
        k = k if k is not None else self.k

        results = self._client.search(
            query,
            top_k=k,
            namespace=self._namespace,
            **kwargs,
        )

        return [
            dotdict(
                long_text=result.text or "",
                id=result.id,
                score=result.score,
                metadata=result.metadata,
            )
            for result in results
        ]
