#!/usr/bin/env python3
import argparse
import asyncio
import logging
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

import datasets
import prompts
from eval import evaluate_dataset
from eval.request import RequestParameters
from loaders import Sample
from loaders.bbq import BBQLoader
from loaders.law import LawLoader
from loaders.winogender import WinogenderLoader

DATASET_NAMES = [
    # Dataclasses mess with mypy's ability to detect attributes
    loader.dataset  # type: ignore[attr-defined]
    for loader in (BBQLoader, LawLoader, WinogenderLoader)
]
PROMPTS = {
    "question": prompts.prompt_question,
    "instruction": prompts.prompt_instruction_following,
}


def configure_logging(verbose: bool) -> None:
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)
    # Reduce OpenAI logging
    logging.getLogger("openai").setLevel(logging.WARNING)


def load_dataset(dataset_name: str) -> Iterable[Sample[Any]]:
    """Load a dataset by name"""
    if dataset_name == BBQLoader.dataset:
        return BBQLoader(datasets.find_bbq_dataset())
    if dataset_name == LawLoader.dataset:
        paths = datasets.find_law_dataset()
        # For the law dataset, generate samples twice: once with race set to
        # "Black" and again with race set to "White".
        return chain.from_iterable(
            LawLoader(paths, parameter_overrides={"race": race})
            for race in ("Black", "White")
        )
    if dataset_name == WinogenderLoader.dataset:
        loader = WinogenderLoader(datasets.find_winogender_dataset())
        loader.load_bls_data(datasets.find_winogender_stats())
        return loader
    raise ValueError(f"Unrecognized dataset name '{dataset_name}'")


async def main() -> None:
    """Evaluate a dataset with a particular prompt style"""
    parser = argparse.ArgumentParser(
        description="Evaluate a dataset with a particular prompt style",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        choices=DATASET_NAMES,
        required=True,
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        choices=PROMPTS.keys(),
        required=True,
        help="Prompt format to use",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="Path where results will be saved",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="""
            Model to use. Must support /chat/completions endpoint
            (default: gpt-3.5-turbo)
        """,
    )
    parser.add_argument(
        "--request-rate-limit",
        type=float,
        default=60.0,
        help="Max API requests per minute (default: 60)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model completions (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Token limit for model completions (default: 32)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds API requests (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of workers to use (default: 16)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)
    loader = load_dataset(args.dataset)
    prompt_func = PROMPTS[args.prompt]
    request_parameters = RequestParameters(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.request_timeout,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    await evaluate_dataset(
        samples=loader,
        prompt_func=prompt_func,
        results_file=args.output_file,
        parameters=request_parameters,
        max_requests_per_min=args.request_rate_limit,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    asyncio.run(main())
