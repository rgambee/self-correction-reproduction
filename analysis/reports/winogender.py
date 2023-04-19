#!/usr/bin/env python3
from analysis.graders.winogender import is_answer_correct
from analysis.metrics import calculate_accuracy
from loaders.winogender import WinogenderParameters

from . import load_results, parse_args


def main() -> None:
    user_args = parse_args()
    for path in user_args.result_paths:
        accuracy = calculate_accuracy(
            is_answer_correct(result)
            for result in load_results(path, WinogenderParameters)
        )
        print(f"{accuracy:6.1%} accuracy for results {path.name}")


if __name__ == "__main__":
    main()
