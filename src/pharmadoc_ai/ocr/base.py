import time
from typing import Protocol

import numpy as np

from pharmadoc_ai.config import Settings
from pharmadoc_ai.errors import ProcessingTimeout
from pharmadoc_ai.schemas import Word


class OCREngine(Protocol):
    def recognize(self, image: np.ndarray, settings: Settings, deadline: float) -> list[Word]: ...


def remaining(deadline: float, limit: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ProcessingTimeout()
    return min(seconds, limit)


def geometric_order(words: list[Word]) -> list[Word]:
    """Cluster words into horizontal lines. Complex multi-column layouts remain limited."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0])):
        center = (word.bbox[1] + word.bbox[3]) / 2
        tolerance = max(0.003, (word.bbox[3] - word.bbox[1]) * 0.45)
        target = next(
            (
                line
                for line in reversed(lines[-4:])
                if abs(center - (line[0].bbox[1] + line[0].bbox[3]) / 2) < tolerance
            ),
            None,
        )
        if target is None:
            lines.append([word])
        else:
            target.append(word)
    ordered = []
    for index, line in enumerate(lines):
        for word in sorted(line, key=lambda w: w.bbox[0]):
            ordered.append(word.model_copy(update={"line": index}))
    return ordered
