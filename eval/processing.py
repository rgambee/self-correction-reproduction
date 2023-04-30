import asyncio
import json
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path

# Import monotonic() on its own so that it can be mocked during testing
from time import monotonic
from typing import Any, Callable, Container, Iterable

import jsonlines
import openai

from eval.request import Request, RequestParameters
from eval.result import Result
from loaders import P, Sample
from prompts.message import Messages


# pylint: disable-next=too-many-arguments
async def process_samples(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], Messages],
    parameters: RequestParameters,
    requests_queue: asyncio.Queue[Request[P]],
    max_requests_per_min: float,
    previously_saved_samples: Container[int] = frozenset(),
) -> None:
    """Prepare samples for submission to the API

    This function limits the rate at which requests are enqueued to stay below the API's
    limit. It does not enforce a token rate limit.
    """
    if max_requests_per_min <= 0:
        raise ValueError("max_requests_per_min must be greater than 0")

    available_requests = max_requests_per_min
    max_requests_per_sec = max_requests_per_min / 60.0
    last_check_time = monotonic()
    for sample in samples:
        # If we've already evaluated this sample, skip it
        if sample.id in previously_saved_samples:
            continue

        # Limit the rate at which requests are enqueued to avoid exceeding the API's
        # limit. It would be more accurate to enforce this limit when the requests are
        # actually submitted. But that's more complex since submissions are spread
        # across several workers, which would need to share state.
        # This is a do-while loop since we want to update available_requests on every
        # iteration of the outer for loop.
        while True:
            now = monotonic()
            available_requests = min(
                available_requests + (now - last_check_time) * max_requests_per_sec,
                max_requests_per_min,
            )
            last_check_time = now
            if available_requests >= 1.0:
                break
            await asyncio.sleep(min(1.0 / max_requests_per_sec, 0.1))
        messages = prompt_func(sample)
        request = Request(parameters=parameters, messages=messages, sample=sample)
        await requests_queue.put(request)
        available_requests = max(available_requests - 1.0, 0.0)


async def process_requests(
    requests_queue: asyncio.Queue[Request[P]],
    results_queue: asyncio.Queue[Result[P]],
    rate_limit_sleep: float = 10.0,
) -> None:
    """Submit requests to the OpenAI API

    Requests are taken from request_queue. Results are pushed to result_queue.
    Each result consists of
        * The original sample used to generate the prompt
        * The reply received from the model

    This coroutine runs until it is canceled.
    """
    logger = logging.getLogger(__name__)
    while True:
        request = await requests_queue.get()
        logger.debug("Submitting request for sample %d", request.sample.id)
        try:
            reply = await request.submit()
        except openai.error.RateLimitError:
            logger.warning("Rate limit exceeded, sleeping before retrying...")
            await requests_queue.put(request)
            await asyncio.sleep(rate_limit_sleep)
            continue
        except openai.error.APIError as err:
            if 500 <= err.http_status < 600:
                logger.debug("Encountered server error:")
                logger.debug("    %s", err.http_body)
                logger.debug("Retrying...")
                await requests_queue.put(request)
                continue
            raise
        else:
            await results_queue.put(
                Result(
                    sample=request.sample,
                    prompt_messages=request.messages,
                    reply=reply,
                )
            )
            requests_queue.task_done()


async def process_intermediate_results(
    intermediate_results_queue: asyncio.Queue[Result[P]],
    requests_queue: asyncio.Queue[Request[P]],
    prompt_func: Callable[[Sample[P], str], Messages],
    parameters: RequestParameters,
) -> None:
    """Turn results back into requests for further evaluation"""
    while True:
        result = await intermediate_results_queue.get()
        sample = result.sample
        model_reasoning = result.reply.choices[0].message.content
        messages = prompt_func(sample, model_reasoning)
        request = Request(parameters=parameters, messages=messages, sample=sample)
        await requests_queue.put(request)
        intermediate_results_queue.task_done()


async def save_results_to_file(
    results_queue: asyncio.Queue[Result[P]],
    results_file: Path,
) -> None:
    """Save results to a file for later analysis

    Each result consists of
        * The original sample used to generate the prompt
        * The reply received from the model

    This coroutine runs until it is canceled.
    """
    logger = logging.getLogger(__name__)
    with jsonlines.open(
        results_file,
        mode="a",
        dumps=partial(json.dumps, default=to_json_serializable_type),
    ) as output:
        while True:
            result = await results_queue.get()
            output.write(result)
            logger.debug("Saved result for sample %d", result.sample.id)
            results_queue.task_done()


def to_json_serializable_type(value: Any) -> Any:
    """Convert a value to a JSON-serializable type"""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
