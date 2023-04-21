#!/usr/bin/env python3
from itertools import chain
from typing import Dict, List

from analysis.graders.assessment import Assessment
from analysis.graders.law import is_admission_recommended
from analysis.metrics import calculate_accuracy, calculate_accuracy_ci
from loaders.law import LawParameters

from . import load_results, parse_args


def main() -> None:
    """Report metrics for results from the law school dataset"""
    user_args = parse_args()
    for path in user_args.result_paths:
        assessments_by_race: Dict[str, List[Assessment]] = {}
        for result in load_results(path, LawParameters):
            assessment = is_admission_recommended(result)
            race = result.sample.parameters.race
            if race not in assessments_by_race:
                assessments_by_race[race] = []
            assessments_by_race[race].append(assessment)

        print("Results for file", path.name)
        overall_admission_rate = calculate_accuracy(
            chain.from_iterable(assessments_by_race.values())
        )
        ar_low, ar_high = calculate_accuracy_ci(
            chain.from_iterable(assessments_by_race.values()),
            confidence=user_args.confidence_level,
        )
        print(
            f"{overall_admission_rate:6.1%} admission rate",
            f"({user_args.confidence_level:3.0%} CI: "
            f"{ar_low:6.1%} - {ar_high:6.1%})",
            "overall",
        )
        for race, assessments in assessments_by_race.items():
            admission_rate = calculate_accuracy(assessments)
            ar_low, ar_high = calculate_accuracy_ci(
                assessments,
                confidence=user_args.confidence_level,
            )
            print(
                f"{admission_rate:6.1%} admission rate",
                f"({user_args.confidence_level:3.0%} CI: {ar_low:6.1%} - {ar_high:6.1%})",
                f"with race={race}",
            )


if __name__ == "__main__":
    main()
