from collections import Counter
from typing import Iterable

from .graders.assessment import Assessment


def calculate_accuracy(assessments: Iterable[Assessment]) -> float:
    counts = Counter(assessments)
    return counts[Assessment.CORRECT] / (
        counts[Assessment.CORRECT] + counts[Assessment.INCORRECT]
    )
