"""Transform pipeline applied to documents on write / update.

Lets users mutate text, vector, or metadata before it lands in the stores —
for enrichment, PII redaction, normalization, or deriving vectors elsewhere.
An optional :class:`LambdaTransform` runs the transform in **the user's own AWS
Lambda**, keeping custom logic in-account (compliance-friendly) and off the
client host.

Transforms are plain callables ``(TransformContext) -> TransformContext``, so
they compose with ordinary Python (closures, ``functools.partial``, lambdas).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

Vector = list[float]
Metadata = dict[str, Any]


@dataclass
class TransformContext:
    """Mutable carrier passed through the pipeline for one document."""

    id: str
    text: Optional[str] = None
    vector: Optional[Vector] = None
    metadata: Metadata = field(default_factory=dict)
    namespace: str = "default"
    op: str = "upsert"  # "upsert" | "update"


Transform = Callable[[TransformContext], TransformContext]


class TransformPipeline:
    """Ordered chain of transforms. Callable and iterable."""

    def __init__(self, transforms: Optional[list[Transform]] = None) -> None:
        self._transforms: list[Transform] = list(transforms or [])

    def add(self, transform: Transform) -> "TransformPipeline":
        self._transforms.append(transform)
        return self

    def __iter__(self):
        return iter(self._transforms)

    def __len__(self) -> int:
        return len(self._transforms)

    def __call__(self, ctx: TransformContext) -> TransformContext:
        for t in self._transforms:
            ctx = t(ctx)
        return ctx


def as_pipeline(spec) -> Optional[TransformPipeline]:
    """Coerce ``None`` / a single callable / a list into a pipeline."""
    if spec is None:
        return None
    if isinstance(spec, TransformPipeline):
        return spec
    if callable(spec):
        return TransformPipeline([spec])
    return TransformPipeline(list(spec))


class LambdaTransform:
    """Invoke a user-owned AWS Lambda to transform a document.

    The Lambda receives ``{"id","text","vector","metadata","namespace","op"}``
    as JSON and must return the same shape (any subset it wants to change).
    """

    def __init__(self, function_name: str, session, qualifier: Optional[str] = None) -> None:
        self._client = session.client("lambda")
        self._function_name = function_name
        self._qualifier = qualifier

    def __call__(self, ctx: TransformContext) -> TransformContext:
        payload = {
            "id": ctx.id,
            "text": ctx.text,
            "vector": ctx.vector,
            "metadata": ctx.metadata,
            "namespace": ctx.namespace,
            "op": ctx.op,
        }
        kwargs = {
            "FunctionName": self._function_name,
            "InvocationType": "RequestResponse",
            "Payload": json.dumps(payload).encode("utf-8"),
        }
        if self._qualifier:
            kwargs["Qualifier"] = self._qualifier
        resp = self._client.invoke(**kwargs)
        body = json.loads(resp["Payload"].read() or b"{}")
        # Lambda may return a subset; only overwrite what it provides.
        ctx.text = body.get("text", ctx.text)
        ctx.vector = body.get("vector", ctx.vector)
        if "metadata" in body and body["metadata"] is not None:
            ctx.metadata = body["metadata"]
        return ctx
