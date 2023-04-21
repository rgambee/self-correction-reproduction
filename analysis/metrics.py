from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Tuple

from scipy import stats

from .graders.assessment import Assessment


@dataclass
class ValueWithConfidence:
    """An value with an associated confidence interval

    The interval is an absolute one:
        interval_low <= value <= interval_high

    An relative interval can be obtained with relative_interval().
    """

    value: float
    interval_low: float
    interval_high: float
    confidence_level: float

    def relative_interval(self) -> Tuple[float, float]:
        return (
            self.interval_low - self.value,
            self.interval_high - self.value,
        )

    def __str__(self) -> str:
        return f"{self.value:6.1%}"

    def __repr__(self) -> str:
        return " ".join(
            (
                f"{self.value:6.1%}",
                f"({self.confidence_level:3.0%} CI:",
                f"{self.interval_low:6.1%} - {self.interval_high:6.1%})",
            )
        )


def calculate_accuracy(
    assessments: Iterable[Assessment],
    confidence_level: float = 0.95,
) -> ValueWithConfidence:
    """Compute the fraction of assessments that are correct

    The returned object is a tuple containing the proportion of correct answers and the
    surrounding confidence interval.

    Assessments of UNKOWN are excluded.
    """
    counts = Counter(assessments)
    result = stats.binomtest(
        k=counts[Assessment.CORRECT],
        n=counts[Assessment.CORRECT] + counts[Assessment.INCORRECT],
    )
    confidence_interval = result.proportion_ci(confidence_level=confidence_level)
    return ValueWithConfidence(
        value=result.proportion_estimate,
        interval_low=confidence_interval[0],
        interval_high=confidence_interval[1],
        confidence_level=confidence_level,
    )
