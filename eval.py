import asyncio
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, Optional, Sequence

import jsonlines
import openai

from loaders import P, Sample


@dataclass
class Completion:
    """A single completion from the OpenAI API"""

    text: str
    index: int
    finish_reason: str
    logprobs: Optional[Dict[str, float]] = None


@dataclass
class Reply:
    """A reply from the OpenAI API for a single sample

    The reply could contain several completions.
    """

    # pylint doesn't like two letter names, claiming they don't conform to the
    # snake_case convention
    id: str  # pylint: disable=invalid-name
    object: str
    created: int
    model: str
    choices: Sequence[Completion]

    def __post_init__(self) -> None:
        self.choices = [
            Completion(**choice) for choice in self.choices  # type: ignore[arg-type]
        ]


@dataclass
class RequestParameters:
    model: str
    max_tokens: int
    temperature: float
    logprobs: int
    timeout: Optional[float] = None


@dataclass
class Request(Generic[P]):
    """A request to the OpenAI API for a single sample"""

    parameters: RequestParameters
    prompt: str
    sample: Sample[P]

    async def submit(self) -> Reply:
        """Submit this request to the OpenAI API and return the reply"""
        resp = await openai.Completion.acreate(  # type: ignore[no-untyped-call]
            prompt=self.prompt, **asdict(self.parameters)
        )
        return Reply(**resp)


@dataclass
class Result(Generic[P]):
    """A combined sample and reply"""

    sample: Sample[P]
    reply: Reply


async def evaluate_dataset(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], str],
    results_file: Path,
    parameters: RequestParameters,
    num_workers: int = 16,
) -> None:
    """Evaluate each sample of a dataset using the OpenAI API

    Results will be appended to the given path.

    This function will skip any samples that have already been evaluated by examining
    the results file.
    """
    # Check the results file to see if we've already evaluated some of the samples
    last_sample_id = find_most_recent_sample(results_file)

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


async def process_samples(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], str],
    parameters: RequestParameters,
    requests_queue: asyncio.Queue[Request[P]],
    last_sample_id: Optional[int] = None,
) -> None:
    """Prepare samples for submission to the API"""
    for sample in samples:
        # If we've already evaluated this sample, skip it
        if last_sample_id is not None and sample.id <= last_sample_id:
            continue

        prompt = prompt_func(sample)
        request = Request(parameters=parameters, prompt=prompt, sample=sample)
        await requests_queue.put(request)


async def process_requests(
    requests_queue: asyncio.Queue[Request[P]],
    results_queue: asyncio.Queue[Result[P]],
    exit_event: asyncio.Event,
    rate_limit_sleep: float = 10.0,
) -> None:
    """Async worker for submitting requests to the OpenAI API

    Requests are taken from request_queue. Results are pushed to result_queue.

    This function runs until exit_event is set.
    """
    while not exit_event.is_set():
        try:
            request = requests_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)
            continue
        try:
            reply = await request.submit()
        except openai.error.RateLimitError:
            await requests_queue.put(request)
            await asyncio.sleep(rate_limit_sleep)
            continue
        else:
            requests_queue.task_done()
            await results_queue.put(Result(sample=request.sample, reply=reply))


async def process_results(
    results_queue: asyncio.Queue[Result[P]],
    results_file: Path,
    exit_event: asyncio.Event,
) -> None:
    """Async worker for saving results to a file

    The function runs until exit_event is set.
    """
    with jsonlines.open(
        results_file,
        mode="a",
        dumps=partial(json.dumps, default=to_json_serializable_type),
    ) as output:
        while not exit_event.is_set():
            try:
                result = results_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
                continue
            output.write(result)
            results_queue.task_done()


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


def to_json_serializable_type(value: Any) -> Any:
    """Convert a value to a JSON-serializable type"""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
