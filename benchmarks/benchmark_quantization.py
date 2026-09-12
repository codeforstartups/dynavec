"""Benchmark ScalarQuantizer vs ProductQuantizer."""

import time

import numpy as np

from dynavec.quantization import ProductQuantizer, ScalarQuantizer


def generate_dataset(n_vectors=10_000, dim=32, seed=0):
    """Generate a reproducible clustered float32 dataset."""
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(8, dim)).astype(np.float32)
    assign = rng.integers(0, 8, size=n_vectors)

    x = centers[assign] + 0.05 * rng.normal(
        size=(n_vectors, dim)
    ).astype(np.float32)

    return x.astype(np.float32)


def benchmark_encode(quantizer, vectors, iterations=5):
    """Measure average encode time."""
    # Warm-up
    quantizer.encode(vectors)

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        quantizer.encode(vectors)
        times.append(time.perf_counter() - start)

    return np.mean(times)


def benchmark_decode(quantizer, codes, iterations=5):
    """Measure average decode time."""
    # Warm-up
    quantizer.decode(codes)

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        quantizer.decode(codes)
        times.append(time.perf_counter() - start)

    return np.mean(times)


def benchmark():
    dimensions = 32
    dataset_sizes = [1_000, 10_000]

    print("=" * 80)
    print("Dynavec Quantization Benchmark")
    print("ScalarQuantizer INT8 vs ProductQuantizer")
    print("=" * 80)

    print()

    for n_vectors in dataset_sizes:
        print(f"Dataset: {n_vectors:,} vectors × {dimensions} dimensions")
        print("-" * 80)

        vectors = generate_dataset(
            n_vectors=n_vectors,
            dim=dimensions,
        )

        raw_bytes_per_vector = dimensions * 4

        # ---------------------------------------------------------
        # Scalar Quantizer
        # ---------------------------------------------------------
        scalar = ScalarQuantizer()

        start = time.perf_counter()
        scalar.fit(vectors)
        scalar_fit_time = time.perf_counter() - start

        scalar_codes = scalar.encode(vectors)

        scalar_encode_time = benchmark_encode(
            scalar,
            vectors,
        )

        scalar_decode_time = benchmark_decode(
            scalar,
            scalar_codes,
        )

        scalar_reconstruction_error = scalar.reconstruction_error(
            vectors
        )

        scalar_bytes_per_vector = scalar.code_size_bytes
        scalar_total_bytes = scalar_codes.nbytes
        scalar_compression = (
            raw_bytes_per_vector / scalar_bytes_per_vector
        )

        # ---------------------------------------------------------
        # Product Quantizer
        # ---------------------------------------------------------
        pq = ProductQuantizer(
            m=8,
            nbits=8,
            iters=25,
            seed=0,
        )

        start = time.perf_counter()
        pq.fit(vectors)
        pq_fit_time = time.perf_counter() - start

        pq_codes = pq.encode(vectors)

        pq_encode_time = benchmark_encode(
            pq,
            vectors,
        )

        pq_decode_time = benchmark_decode(
            pq,
            pq_codes,
        )

        pq_reconstruction_error = pq.reconstruction_error(
            vectors
        )

        pq_bytes_per_vector = pq.code_size_bytes
        pq_total_bytes = pq_codes.nbytes
        pq_compression = (
            raw_bytes_per_vector / pq_bytes_per_vector
        )

        # ---------------------------------------------------------
        # Results
        # ---------------------------------------------------------
        print()
        print("ScalarQuantizer (INT8)")
        print(f"  Fit time:                  {scalar_fit_time:.4f} sec")
        print(f"  Encode time:               {scalar_encode_time:.4f} sec")
        print(f"  Decode time:               {scalar_decode_time:.4f} sec")
        print(
            f"  Reconstruction error:     "
            f"{scalar_reconstruction_error:.8f}"
        )
        print(f"  Bytes/vector:              {scalar_bytes_per_vector}")
        print(f"  Total encoded size:        {scalar_total_bytes / 1024:.2f} KB")
        print(f"  Compression:               {scalar_compression:.2f}x")

        print()
        print("ProductQuantizer (PQ)")
        print(f"  Fit time:                  {pq_fit_time:.4f} sec")
        print(f"  Encode time:               {pq_encode_time:.4f} sec")
        print(f"  Decode time:               {pq_decode_time:.4f} sec")
        print(
            f"  Reconstruction error:     "
            f"{pq_reconstruction_error:.8f}"
        )
        print(f"  Bytes/vector:              {pq_bytes_per_vector}")
        print(f"  Total encoded size:        {pq_total_bytes / 1024:.2f} KB")
        print(f"  Compression:               {pq_compression:.2f}x")

        print()
        print("Comparison")
        print(
            f"  Scalar/PQ compression:     "
            f"{scalar_compression / pq_compression:.2f}x"
        )

        if scalar_encode_time > 0:
            print(
                f"  PQ encode / Scalar encode: "
                f"{pq_encode_time / scalar_encode_time:.2f}x"
            )

        if scalar_decode_time > 0:
            print(
                f"  PQ decode / Scalar decode: "
                f"{pq_decode_time / scalar_decode_time:.2f}x"
            )

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    benchmark()
