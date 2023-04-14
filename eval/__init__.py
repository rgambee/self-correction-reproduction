import asyncio
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import jsonlines

from eval.classes import Request, RequestParameters, Result
from eval.processing import process_requests, process_results, process_samples
from loaders import P, Sample


# pylint: disable-next=too-many-arguments
async def evaluate_dataset(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], str],
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
    last_sample_id = find_most_recent_sample(results_file)
    if last_sample_id is not None:
        logger.info("Resuming from sample %d", last_sample_id)

    requests_queue: asyncio.Queue[Request[P]] = asyncio.Queue(maxsize=num_workers)
    results_queue: asyncio.Queue[Result[P]] = asyncio.Queue(maxsize=num_workers)
    exit_event = asyncio.Event()

    # Create tasks to track all the workers
    sample_worker = asyncio.create_task(
        process_samples(
            samples=samples,
            prompt_func=prompt_func,
            parameters=parameters,
            requests_queue=requests_queue,
            max_requests_per_min=max_requests_per_min,
            last_sample_id=last_sample_id,
        ),
    )
    request_workers = [
        asyncio.create_task(
            process_requests(
                requests_queue=requests_queue,
                results_queue=results_queue,
                exit_event=exit_event,
            ),
        )
        for _ in range(num_workers)
    ]
    result_worker = asyncio.create_task(
        process_results(
            results_queue=results_queue,
            results_file=results_file,
            exit_event=exit_event,
        ),
    )

    # Wait for all samples to be processed
    await sample_worker
    # Wait for requests to be sent and results saved
    await requests_queue.join()
    await results_queue.join()
    # Tell workers to exit
    exit_event.set()
    # Wait for them to return
    await asyncio.wait(request_workers + [result_worker])


def find_most_recent_sample(results_file: Path) -> Optional[int]:
    """Check the results file to get the id of the sample that was last evaluated

    This is helpful for resuming a long evaluation after it's been interrupted.

    If the results file doesn't exist or doesn't contain any valid results, return None.

    This function assumes the results are sorted by sample id.
    """
    if not results_file.exists():
        return None

    last_id: Optional[int] = None
    with jsonlines.open(results_file) as reader:
        for line in reader.iter(type=dict, skip_invalid=True):
            try:
                last_id = int(line.get("sample", {}).get("id", None))
            except (AttributeError, TypeError, ValueError):
                pass
    return last_id
