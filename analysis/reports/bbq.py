#!/usr/bin/env python3
from analysis.graders.bbq import is_answer_correct
from analysis.metrics import calculate_accuracy
from analysis.reports import load_results, parse_args
from loaders.bbq import BBQParameters


def main() -> None:
    """Report metrics for results from the BBQ dataset"""
    user_args = parse_args()
    for path in user_args.result_paths:
        assessments = (
            is_answer_correct(result) for result in load_results(path, BBQParameters)
        )
        accuracy = calculate_accuracy(
            assessments, confidence_level=user_args.confidence_level
        )

        print("Results for file", path.name)
        print(f"{accuracy!r} accuracy overall")


if __name__ == "__main__":
    main()
