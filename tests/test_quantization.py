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


def test_save_before_fit_raises(tmp_path):
    pq = ProductQuantizer(m=4)
    with pytest.raises(RuntimeError, match="must be .fit\\(\\) before use"):
        pq.save(tmp_path / "model.npz")


def test_round_trip_save_and_load_file(tmp_path, clustered):
    pq = ProductQuantizer(m=8, nbits=8, iters=15, seed=42).fit(clustered)
    orig_codes = pq.encode(clustered)
    orig_recon = pq.decode(orig_codes)
    orig_error = pq.reconstruction_error(clustered)
    orig_dists = pq.asymmetric_distances(clustered[0], orig_codes)

    file_path = tmp_path / "quantizer.npz"
    pq.save(file_path)

    loaded = ProductQuantizer.load(file_path)
    assert loaded.is_fitted
    assert loaded.m == 8
    assert loaded.nbits == 8
    assert loaded.iters == 15
    assert loaded.seed == 42
    assert loaded._dsub == pq._dsub
    np.testing.assert_array_almost_equal(loaded._codebooks, pq._codebooks)

    loaded_codes = loaded.encode(clustered)
    np.testing.assert_array_equal(loaded_codes, orig_codes)

    loaded_recon = loaded.decode(loaded_codes)
    np.testing.assert_array_almost_equal(loaded_recon, orig_recon)

    assert loaded.reconstruction_error(clustered) == pytest.approx(orig_error)

    loaded_dists = loaded.asymmetric_distances(clustered[0], loaded_codes)
    np.testing.assert_array_almost_equal(loaded_dists, orig_dists)


def test_round_trip_save_and_load_bytesio(clustered):
    import io

    pq = ProductQuantizer(m=4, nbits=8, iters=10, seed=7).fit(clustered)
    buf = io.BytesIO()
    pq.save(buf)
    buf.seek(0)

    loaded = ProductQuantizer.load(buf)
    assert loaded.is_fitted
    assert loaded.m == 4
    np.testing.assert_array_equal(loaded.encode(clustered), pq.encode(clustered))


def test_load_invalid_or_missing_version_raises(tmp_path):
    bad_version = tmp_path / "bad_version.npz"
    np.savez(
        bad_version,
        format_version=np.array(999, dtype=np.int32),
        m=np.array(4, dtype=np.int32),
        nbits=np.array(8, dtype=np.int32),
        iters=np.array(10, dtype=np.int32),
        seed=np.array(0, dtype=np.int32),
        dsub=np.array(8, dtype=np.int32),
        codebooks=np.zeros((4, 256, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Unsupported quantizer format version: 999"):
        ProductQuantizer.load(bad_version)

    no_version = tmp_path / "no_version.npz"
    np.savez(no_version, m=np.array(4, dtype=np.int32))
    with pytest.raises(ValueError, match="missing format_version header"):
        ProductQuantizer.load(no_version)


def test_load_missing_field_raises(tmp_path):
    missing_codebooks = tmp_path / "missing_codebooks.npz"
    np.savez(
        missing_codebooks,
        format_version=np.array(1, dtype=np.int32),
        m=np.array(4, dtype=np.int32),
        nbits=np.array(8, dtype=np.int32),
        iters=np.array(10, dtype=np.int32),
        seed=np.array(0, dtype=np.int32),
        dsub=np.array(8, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="missing 'codebooks' field"):
        ProductQuantizer.load(missing_codebooks)


def test_load_corrupted_file_raises(tmp_path):
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a valid npz archive")
    with pytest.raises(ValueError):
        ProductQuantizer.load(corrupt)

