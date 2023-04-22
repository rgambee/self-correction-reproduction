from collections import Counter
from typing import Iterable

from scipy import stats

from .graders.assessment import Assessment


class BinomialDistribution:
    """A binomial distribution with an associated confidence interval

    The interval is an absolute one:
        interval_low <= value <= interval_high

    An relative interval can be obtained with ci_low_rel and ci_high_rel.
    """

    def __init__(
        self,
        successes: int,
        samples: int,
        confidence_level: float,
    ) -> None:
        self.binom_test_result = stats.binomtest(successes, samples)
        self.confidence_level = confidence_level
        self.confidence_interval = self.binom_test_result.proportion_ci(
            confidence_level=self.confidence_level,
            method="exact",
        )

    @property
    def proportion(self) -> float:
        return float(self.binom_test_result.statistic)

    @property
    def ci_low(self) -> float:
        return float(self.confidence_interval.low)

    @property
    def ci_high(self) -> float:
        return float(self.confidence_interval.high)

    @property
    def ci_low_rel(self) -> float:
        return self.ci_low - self.proportion

    @property
    def ci_high_rel(self) -> float:
        return self.ci_high - self.proportion

    def __str__(self) -> str:
        return f"{self.proportion:6.1%}"

    def __repr__(self) -> str:
        return " ".join(
            (
                f"{self.proportion:6.1%}",
                f"({self.confidence_level:3.0%} CI:",
                f"{self.ci_low:6.1%} - {self.ci_high:6.1%})",
            )
        )


def calculate_accuracy(
    assessments: Iterable[Assessment],
    confidence_level: float = 0.95,
) -> BinomialDistribution:
    """Compute the fraction of assessments that are correct

    The returned object is a tuple containing the proportion of correct answers and the
    surrounding confidence interval.

    Assessments of UNKOWN are excluded.
    """
    counts = Counter(assessments)
    return BinomialDistribution(
        successes=counts[Assessment.CORRECT],
        samples=counts[Assessment.CORRECT] + counts[Assessment.INCORRECT],
        confidence_level=confidence_level,
    )
