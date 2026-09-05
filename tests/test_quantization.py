"""Tests for Product Quantization (pure, no AWS)."""

import numpy as np
import pytest

from dynavec.quantization import ProductQuantizer


@pytest.fixture
def clustered():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(8, 32)).astype(np.float32)
    assign = rng.integers(0, 8, size=500)
    x = centers[assign] + 0.05 * rng.normal(size=(500, 32)).astype(np.float32)
    return x.astype(np.float32)


def test_fit_encode_shapes(clustered):
    pq = ProductQuantizer(m=8, nbits=8).fit(clustered)
    codes = pq.encode(clustered)
    assert codes.shape == (500, 8)
    assert codes.dtype == np.uint8


def test_code_size_and_compression(clustered):
    pq = ProductQuantizer(m=8, nbits=8).fit(clustered)
    # 32 dims * 4 bytes = 128 bytes -> 8 bytes codes = 16x compression
    assert pq.code_size_bytes == 8
    raw = clustered.shape[1] * 4
    assert raw / pq.code_size_bytes == 16


def test_m_must_divide_dimension(clustered):
    with pytest.raises(ValueError):
        ProductQuantizer(m=5).fit(clustered)  # 32 % 5 != 0


def test_reconstruction_error_is_small_for_clustered(clustered):
    pq = ProductQuantizer(m=8, nbits=8).fit(clustered)
    err = pq.reconstruction_error(clustered)
    # tight clusters -> low reconstruction error
    assert err < 1.0


def test_asymmetric_distance_ranks_self_first(clustered):
    pq = ProductQuantizer(m=8, nbits=8).fit(clustered)
    codes = pq.encode(clustered)
    q = clustered[0]
    dists = pq.asymmetric_distances(q, codes)
    # the encoded version of the query itself should be among the nearest
    assert dists.argmin() < 50  # its own cluster members dominate the front


def test_use_before_fit_raises():
    pq = ProductQuantizer(m=4)
    with pytest.raises(RuntimeError):
        pq.encode(np.zeros((1, 16), dtype=np.float32))
