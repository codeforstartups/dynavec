import pytest

from dynavec import DynavecConfig


def config(**kwargs: object) -> DynavecConfig:
    return DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="table",
        dimension=128,
        **kwargs,
    )


@pytest.mark.parametrize("max_workers", [0, -1, -10, 2.5, "4", True, False, None])
def test_max_workers_must_be_a_positive_integer(max_workers: object) -> None:
    with pytest.raises(ValueError, match="^max_workers must be a positive integer$"):
        config(max_workers=max_workers)


@pytest.mark.parametrize("max_workers", [1, 4, 16])
def test_positive_max_workers_are_accepted(max_workers: int) -> None:
    assert config(max_workers=max_workers).max_workers == max_workers


def test_max_workers_defaults_to_eight() -> None:
    assert config().max_workers == 8
