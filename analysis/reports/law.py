#!/usr/bin/env python3
from itertools import chain
from typing import Dict, List

from analysis.graders.assessment import Assessment
from analysis.graders.law import is_admission_recommended
from analysis.metrics import calculate_accuracy
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
            chain.from_iterable(assessments_by_race.values()),
            confidence_level=user_args.confidence_level,
        )
        print(f"{overall_admission_rate!r} admission rate overall")
        for race, assessments in assessments_by_race.items():
            admission_rate = calculate_accuracy(
                assessments,
                confidence_level=user_args.confidence_level,
            )
            print(f"{admission_rate!r} admission rate with race={race}")


if __name__ == "__main__":
    main()
