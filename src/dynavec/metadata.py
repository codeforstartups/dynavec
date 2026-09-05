"""Metadata handling: the auto-generate switch and the two-store split.

dynavec keeps a **small filterable subset** of metadata in S3 Vectors (so it can
pre-filter during ANN) and the **full, rich metadata + source text** in DynamoDB
(cheap to read, no per-vector size cap, single-digit-ms hydration).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

from .config import NS_METADATA_KEY, TEXT_METADATA_KEY, DynavecConfig

Metadata = dict[str, Any]

_WORD_RE = re.compile(r"\w+")

# S3 Vectors metadata values must be str | number | bool | list. Anything else
# (dict, None, nested) is kept in DynamoDB only.
_S3_SCALAR = (str, int, float, bool)


def generate_auto_metadata(text: Optional[str]) -> Metadata:
    """Derive lightweight, useful metadata from raw text.

    This is what dynavec attaches when the caller sets ``auto_metadata=True`` and
    does not supply their own. Cheap, deterministic, and handy for filtering
    (e.g. ``created_at``, ``char_count``) and dedup (``content_hash``).
    """
    now = datetime.now(timezone.utc).isoformat()
    meta: Metadata = {"created_at": now}
    if text:
        meta["char_count"] = len(text)
        meta["word_count"] = len(_WORD_RE.findall(text))
        meta["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return meta


def _is_s3_filterable_value(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, int, float)):
        return True
    if isinstance(value, list):
        return all(isinstance(v, _S3_SCALAR) for v in value)
    return False


def split_metadata(
    metadata: Metadata,
    config: DynavecConfig,
    namespace: str,
    text: Optional[str] = None,
) -> tuple[Metadata, Metadata]:
    """Split metadata into (s3_vectors_metadata, dynamodb_metadata).

    - S3 Vectors gets ``config.filterable_keys`` (or all scalar keys if that is
      ``None``), always tagged with the namespace so a shared index can hold many
      namespaces and still filter cleanly.
    - DynamoDB gets the complete, untouched metadata.
    """
    dynamo_meta = dict(metadata)  # full copy, canonical source of truth

    if config.filterable_keys is None:
        candidate_keys = [k for k, v in metadata.items() if _is_s3_filterable_value(v)]
    else:
        candidate_keys = list(config.filterable_keys)

    s3_meta: Metadata = {}
    for key in candidate_keys:
        if key in config.non_filterable_keys:
            continue
        if key in metadata and _is_s3_filterable_value(metadata[key]):
            s3_meta[key] = metadata[key]

    # Namespace tag is always present so we can scope queries within one index.
    s3_meta[NS_METADATA_KEY] = namespace

    if config.store_text_in_s3vectors and text:
        s3_meta[TEXT_METADATA_KEY] = text[: config.text_mirror_max_chars]

    return s3_meta, dynamo_meta


def build_s3_filter(user_filter: Optional[Metadata], namespace: str) -> Metadata:
    """Combine a user metadata filter with the mandatory namespace scope.

    Accepts the S3 Vectors filter dialect (MongoDB-style operators such as
    ``$eq``, ``$gte``, ``$in``, ``$and``, ``$or``). A bare ``{"k": v}`` means
    equality. We AND-in the namespace tag.
    """
    ns_clause = {NS_METADATA_KEY: namespace}
    if not user_filter:
        return ns_clause
    # If the user already used a top-level $and, extend it; otherwise wrap.
    if set(user_filter.keys()) == {"$and"} and isinstance(user_filter["$and"], list):
        return {"$and": [*user_filter["$and"], ns_clause]}
    return {"$and": [user_filter, ns_clause]}
