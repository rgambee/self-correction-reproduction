import asyncio
import logging
from pathlib import Path
from typing import Callable, Iterable, Set

import jsonlines

from eval.pipeline import Pipeline, Stage
from eval.processing import process_requests, process_results, process_samples
from eval.request import Request, RequestParameters
from eval.result import Result
from loaders import P, Sample
from prompts.message import Messages


# pylint: disable-next=too-many-arguments
async def evaluate_dataset(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], Messages],
    results_file: Path,
    parameters: RequestParameters,
    max_requests_per_min: float,
    num_workers: int = 16,
) -> None:
    """Evaluate each sample of a dataset using the OpenAI API

    Results will be appended to the file at the given path.

    This function will skip any samples that have already been evaluated by examining
    the results file. It will enforce a request rate limit but not a token rate limit.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    logger = logging.getLogger(__name__)
    # Check the results file to see if we've already evaluated some of the samples
    saved_samples = get_saved_samples(results_file)
    if saved_samples:
        logger.info("Skipping %d previously evaluated samples", len(saved_samples))

    pipeline = Pipeline()
    requests_queue: asyncio.Queue[Request[P]] = asyncio.Queue(maxsize=num_workers)
    results_queue: asyncio.Queue[Result[P]] = asyncio.Queue(maxsize=num_workers)

    # Create a pipeline to hold all the tasks
    # Stage 0: turn Samples into Requests using prompt_func()
    pipeline.append_stage(
        Stage.from_coro(
            coro_func=process_samples,
            kwargs={
                "samples": samples,
                "prompt_func": prompt_func,
                "parameters": parameters,
                "requests_queue": requests_queue,
                "max_requests_per_min": max_requests_per_min,
                "previously_saved_samples": saved_samples,
            },
            output_queue=requests_queue,
        )
    )
    # Stage 1: turn Requests into Results by submitting them to the API
    pipeline.append_stage(
        Stage.from_coro(
            coro_func=process_requests,
            kwargs={
                "requests_queue": requests_queue,
                "results_queue": results_queue,
            },
            output_queue=results_queue,
            num_tasks=num_workers,
        )
    )
    # Stage 2: save Results to file
    pipeline.append_stage(
        Stage.from_coro(
            coro_func=process_results,
            kwargs={
                "results_queue": results_queue,
                "results_file": results_file,
            },
            output_queue=None,
        )
    )
    await pipeline.wait(timeout=parameters.timeout)


def get_saved_samples(results_file: Path) -> Set[int]:
    """Check the results file to get the IDs of samples that have already been evaluated

    This is helpful for resuming a long evaluation after it's been interrupted.

    If the results file doesn't exist or doesn't contain any valid results, return the
    empty set.
    """
    ids: Set[int] = set()
    if not results_file.exists():
        return ids
    with jsonlines.open(results_file) as reader:
        for line in reader.iter(type=dict, skip_invalid=True):
            try:
                sample_id = int(line.get("sample", {}).get("id", None))
            except (AttributeError, TypeError, ValueError):
                pass
            else:
                if sample_id is not None:
                    ids.add(sample_id)
    return ids
