from collections.abc import Mapping
from typing import Any

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction

class Retrieve(Parameter):
    name: str
    input_variable: str
    desc: str
    stage: str
    k: int
    callbacks: list[Any]

    def __init__(self, k: int = ..., callbacks: list[Any] | None = ...) -> None: ...
    def reset(self) -> None: ...
    def dump_state(self) -> dict[str, int]: ...
    def load_state(self, state: Mapping[str, Any]) -> None: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[str] | Prediction | list[Prediction]: ...
    def forward(
        self, query: str, k: int | None = ..., **kwargs: Any
    ) -> list[str] | Prediction | list[Prediction]: ...
