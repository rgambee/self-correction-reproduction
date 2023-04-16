import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

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
    sample_task = asyncio.create_task(
        process_samples(
            samples=samples,
            prompt_func=prompt_func,
            parameters=parameters,
            requests_queue=requests_queue,
            max_requests_per_min=max_requests_per_min,
            last_sample_id=last_sample_id,
        ),
    )
    sample_task.set_name("process_samples")

    request_tasks: List[asyncio.Task[None]] = []
    for i in range(num_workers):
        task = asyncio.create_task(
            process_requests(
                requests_queue=requests_queue,
                results_queue=results_queue,
                exit_event=exit_event,
            ),
        )
        task.set_name(f"process_requests_{i:02}")
        request_tasks.append(task)

    result_task = asyncio.create_task(
        process_results(
            results_queue=results_queue,
            results_file=results_file,
            exit_event=exit_event,
        ),
    )
    result_task.set_name("process_results")

    pending_tasks = set(request_tasks + [sample_task, result_task])
    finished = False
    try:
        while not finished and pending_tasks:
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                logging.debug("Task %s is done", task.get_name())
                err = task.exception()
                if err is not None:
                    raise err
                if task is sample_task:
                    # No more samples. Break out of while loop to shut down other tasks.
                    logging.info("No more samples, shutting down...")
                    finished = True
    except Exception:
        logging.exception("Encountered error, shutting down...")
        raise
    finally:
        # Shut things down from upstream to downstream to avoid information being lost
        await stop_task(sample_task)
        for task in request_tasks:
            await stop_task(sample_task)
        if not result_task.done():
            # The result task is still running. Give it a moment to finish saving any
            # results still in the queue.
            try:
                await asyncio.wait_for(results_queue.join(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.debug("Not all pending results were able to be saved")
        await stop_task(result_task)


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


async def stop_task(task: asyncio.Task[Any], timeout: float = 1.0) -> None:
    """Stop the given task gracefully by canceling it and waiting for it to finish"""
    logger = logging.getLogger(__name__)
    if not task.done():
        logger.debug("Canceling task %s", task.get_name())
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            logger.debug("Task %s canceled", task.get_name())
        except asyncio.TimeoutError:
            logger.debug("Timed out while waiting for %s to cancel", task.get_name())
