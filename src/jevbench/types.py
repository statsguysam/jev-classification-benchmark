from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    label: int


@dataclass
class PreparedDataset:
    name: str
    labels: list[str]
    train: list[Row]
    validation: list[Row]
    test: list[Row]
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    row_id: str
    label: int | None
    probabilities: list[float] | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Classifier(Protocol):
    def predict(self, row: Row, labels: list[str], prompt: str) -> Prediction: ...
