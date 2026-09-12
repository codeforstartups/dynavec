"""Property-based tests for chunk_text()'s coverage/overlap invariants.

chunk_text() is a sliding-window chunker: consecutive windows advance by
`step = chunk_size - overlap` characters. When the source text contains no
whitespace, every window is non-blank and therefore yielded (the
whitespace-skip branch never fires), which lets us state a strong,
implementation-independent invariant: stitching the chunks back together by
dropping the first `overlap` characters of every chunk after the first must
reproduce the original text exactly. That is precisely "no gaps and no
duplication beyond the configured overlap."

Whitespace-only / mixed-whitespace inputs are covered separately by the
existing example-based tests in test_ingest.py, since the skip-blank-windows
behavior intentionally breaks the clean tiling invariant used here.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dynavec.ingest import chunk_text

# Non-whitespace, non-control text so every sliding window is guaranteed
# non-blank (piece.strip() is always truthy for a non-empty, non-whitespace
# piece), which keeps the tiling invariant below exact.
_non_whitespace_text = st.text(
    alphabet=st.characters(
        min_codepoint=33,
        max_codepoint=0x2FFF,
        blacklist_categories=("Cs", "Cc", "Zs", "Zl", "Zp"),
    ).filter(lambda c: not c.isspace()),
    min_size=1,
    max_size=300,
)


@st.composite
def _text_and_valid_window(draw):
    text = draw(_non_whitespace_text)
    chunk_size = draw(st.integers(min_value=1, max_value=len(text) + 20))
    overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return text, chunk_size, overlap


@given(_text_and_valid_window())
@settings(max_examples=300)
def test_chunk_text_reconstructs_original_with_no_gaps_or_dupes(params):
    """Dropping each chunk's overlapping prefix and concatenating must
    reproduce the source text exactly -> no gaps, no duplication beyond the
    configured overlap."""
    text, chunk_size, overlap = params
    chunks = list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))

    assert chunks, "non-whitespace text must always yield at least one chunk"

    reconstructed = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
    assert reconstructed == text


@given(_text_and_valid_window())
@settings(max_examples=300)
def test_chunk_text_chunks_are_bounded_and_only_last_may_be_short(params):
    """Every chunk is <= chunk_size, and only the final chunk may be shorter
    (the tail truncation at end-of-text)."""
    text, chunk_size, overlap = params
    chunks = list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))

    assert all(len(c) <= chunk_size for c in chunks)
    for c in chunks[:-1]:
        assert len(c) == chunk_size


@given(_text_and_valid_window())
@settings(max_examples=300)
def test_chunk_text_every_chunk_is_exact_substring_at_expected_offset(params):
    """Each chunk must be a verbatim substring of the source, at the offset
    implied by the sliding window step (no off-by-one drift)."""
    text, chunk_size, overlap = params
    chunks = list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))

    step = chunk_size - overlap
    for i, chunk in enumerate(chunks):
        start = i * step
        assert chunk == text[start : start + chunk_size]


@given(
    chunk_size=st.integers(min_value=1, max_value=500),
    overlap=st.integers(min_value=0, max_value=500),
    text=st.text(min_size=0, max_size=50),
)
@settings(max_examples=200)
def test_chunk_text_overlap_must_be_strictly_less_than_chunk_size(chunk_size, overlap, text):
    """Regardless of input text, overlap >= chunk_size always raises."""
    assume(overlap >= chunk_size)

    with pytest.raises(ValueError, match="overlap must be < chunk_size"):
        list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))


@given(
    chunk_size=st.integers(max_value=0),
    overlap=st.integers(min_value=0, max_value=50),
    text=st.text(min_size=0, max_size=50),
)
@settings(max_examples=100)
def test_chunk_text_non_positive_chunk_size_always_raises(chunk_size, overlap, text):
    """Regardless of overlap or text, chunk_size <= 0 always raises."""
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
