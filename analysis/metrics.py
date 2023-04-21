from collections import Counter
from typing import Iterable, Tuple, cast

from scipy import stats

from .graders.assessment import Assessment


def calculate_accuracy(assessments: Iterable[Assessment]) -> float:
    """Compute the fraction of assessments that are correct

    Assessments of UNKOWN are excluded.
    """
    counts = Counter(assessments)
    return counts[Assessment.CORRECT] / (
        counts[Assessment.CORRECT] + counts[Assessment.INCORRECT]
    )


def calculate_accuracy_ci(
    assessments: Iterable[Assessment],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute the confidence interval around the accuracy"""
    counts = Counter(assessments)
    result = stats.binomtest(
        k=counts[Assessment.CORRECT],
        n=counts[Assessment.CORRECT] + counts[Assessment.INCORRECT],
    )
    return cast(
        Tuple[float, float],
        tuple(result.proportion_ci(confidence_level=confidence)),
    )
