#!/usr/bin/env python3
from collections import defaultdict
from itertools import chain
from typing import Dict, List

from matplotlib import pyplot as plt

from analysis.graders.assessment import Assessment
from analysis.graders.law import is_admission_recommended
from analysis.metrics import ValueWithConfidence, calculate_accuracy
from loaders.law import LawParameters

from . import UserArguments, load_results, parse_args


def plot_admission_rates(
    admission_rates: Dict[str, Dict[str, ValueWithConfidence]],
    user_args: UserArguments,
) -> None:
    """Plot admission rates by results file and race using a bar chart"""
    _, axis = plt.subplots()
    width = 1.0 / (len(admission_rates) + 1)

    for i, (race, rates_by_path) in enumerate(admission_rates.items()):
        xoffset = i * width - len(admission_rates) * width / 2.0
        xcoords = [j + xoffset for j in range(len(user_args.result_paths))]
        rates = [rates_by_path[p.name].value * 100.0 for p in user_args.result_paths]
        axis.bar(xcoords, rates, width=width, align="edge", label=f"Race: {race}")

    axis.set_xticks(
        range(len(user_args.result_paths)),
        labels=user_args.result_paths,
    )
    axis.set_ylim(0, 100)
    axis.set_xlabel("Results file")
    axis.set_ylabel("Admission Rate (%)")
    axis.set_title("Admission Rates for Law School Dataset")
    axis.legend()
    plt.show()


def main() -> None:
    """Report metrics for results from the law school dataset"""
    user_args = parse_args()
    admission_rates: Dict[str, Dict[str, ValueWithConfidence]] = {}
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
            rate = calculate_accuracy(
                assessments,
                confidence_level=user_args.confidence_level,
            )
            print(f"{rate!r} admission rate with race={race}")
            if race not in admission_rates:
                admission_rates[race] = defaultdict(
                    lambda: ValueWithConfidence(0.0, 0.0, 0.0, 0.0)
                )
            admission_rates[race][path.name] = rate

    if user_args.plot:
        plot_admission_rates(admission_rates, user_args)


if __name__ == "__main__":
    main()
