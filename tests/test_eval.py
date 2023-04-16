import asyncio
import json
import logging
import os
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Union
from unittest.mock import MagicMock, call, patch

from eval import evaluate_dataset, find_most_recent_sample
from eval.classes import Request, RequestParameters
from eval.processing import process_samples
from loaders.law import LawLoader, LawParameters, LawSample
from prompts import prompt_question
from tests.test_loaders import TestLawLoader
from tests.utils import LAW_SAMPLE, make_temp_file, write_dummy_dataset


def create_mock_params(
    model: str = "davinci",
    max_tokens: int = 256,
    temperature: float = 1.0,
    logprobs: int = 1,
) -> RequestParameters:
    """Create a mock RequestParameters object suitable for testing"""
    return RequestParameters(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )


def create_mock_response(**kwargs: Any) -> Dict[str, Any]:
    """Create a mock API response suitable for testing"""
    response = {
        "id": "cmpl-1234567890",
        "object": "text_completion",
        "created": 1234567890,
        "model": "davinci",
        "choices": [
            {
                "text": "This is a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
    }
    response.update(kwargs)
    return response


@patch("openai.Completion.acreate", return_value=create_mock_response())
class TestDatasetEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_request_submission(self, mock_api: MagicMock) -> None:
        """Test that a request is sent to the API in the proper format"""
        mock_params = create_mock_params()
        mock_prompt = "This is a test"
        request = Request(prompt=mock_prompt, parameters=mock_params, sample=LAW_SAMPLE)
        await request.submit()
        mock_api.assert_called_once_with(prompt=mock_prompt, **asdict(mock_params))

    async def test_response_handling(self, mock_api: MagicMock) -> None:
        """Test that API responses are parsed and saved correctly"""
        mock_params = create_mock_params()
        mock_prompt = "This is a test"
        mock_sample = LAW_SAMPLE
        with make_temp_file() as temp_output:
            await evaluate_dataset(
                samples=[mock_sample],
                prompt_func=lambda s: mock_prompt,
                results_file=temp_output,
                parameters=mock_params,
                max_requests_per_min=100.0,
                num_workers=1,
            )
            with open(temp_output, encoding="utf-8") as file:
                results = json.load(file)

        self.assertIn("sample", results)
        self.assertEqual(results["sample"], asdict(mock_sample))
        self.assertIn("reply", results)
        self.assertEqual(results["reply"], mock_api.return_value)

    async def test_response_error(self, mock_api: MagicMock) -> None:
        """Test that API errors propagated to caller"""
        mock_api.return_value = {"message": "Invalid reply"}
        with self.assertLogs(level=logging.ERROR):
            with self.assertRaises(TypeError):
                await evaluate_dataset(
                    samples=[LAW_SAMPLE],
                    prompt_func=prompt_question,
                    results_file=Path(os.devnull),
                    parameters=create_mock_params(),
                    max_requests_per_min=100.0,
                    num_workers=1,
                )

        mock_api.side_effect = RuntimeError("Invalid request")
        with self.assertLogs(level=logging.ERROR):
            with self.assertRaises(RuntimeError):
                await evaluate_dataset(
                    samples=[LAW_SAMPLE],
                    prompt_func=prompt_question,
                    results_file=Path(os.devnull),
                    parameters=create_mock_params(),
                    max_requests_per_min=100.0,
                    num_workers=1,
                )

    async def test_end_to_end(self, mock_api: MagicMock) -> None:
        """Test that samples are loaded, requests are sent and replies saved"""
        mock_params = create_mock_params()
        mock_prompt = MagicMock(side_effect=prompt_question)
        # Use dummy law school dataset since that contains multiple samples
        with write_dummy_dataset(TestLawLoader.DUMMY_DATA) as temp_input:
            loader = LawLoader(temp_input)
            samples = list(loader)
            with make_temp_file() as temp_output:
                await evaluate_dataset(
                    samples=samples,
                    prompt_func=mock_prompt,
                    results_file=temp_output,
                    parameters=mock_params,
                    max_requests_per_min=100.0,
                    num_workers=len(samples),
                )
                self.assertEqual(
                    mock_prompt.mock_calls,
                    [call(samp) for samp in samples],
                )
                self.assertEqual(
                    mock_api.mock_calls,
                    [
                        call(
                            prompt=prompt_question(samp),
                            **asdict(mock_params),
                        )
                        for samp in samples
                    ],
                )
                result_index = 0
                with open(temp_output, encoding="utf-8") as file:
                    for line in file:
                        result = json.loads(line)
                        self.assertIn("sample", result)
                        result["sample"]["parameters"] = LawParameters(
                            **result["sample"]["parameters"]
                        )
                        loaded_sample = LawSample(**result["sample"])
                        self.assertEqual(loaded_sample, samples[result_index])
                        self.assertIn("prompt", result)
                        self.assertEqual(
                            result["prompt"], prompt_question(loaded_sample)
                        )
                        self.assertIn("reply", result)
                        self.assertEqual(result["reply"], mock_api.return_value)
                        result_index += 1
        self.assertEqual(result_index, len(samples))

    async def test_resume(self, mock_api: MagicMock) -> None:
        """Test that the dataset is resumed from the last sample"""
        mock_params = create_mock_params()
        mock_prompt = "This is a test"
        with write_dummy_dataset(TestLawLoader.DUMMY_DATA) as temp_input:
            loader = LawLoader(temp_input)
            samples = list(loader)
            with make_temp_file() as temp_output:
                with open(temp_output, mode="w", encoding="utf-8") as file:
                    # Pretend the first sample is already in the results file
                    json.dump({"sample": {"id": 0}}, file)
                    file.write("\n")
                await evaluate_dataset(
                    samples=samples,
                    prompt_func=lambda s: mock_prompt,
                    results_file=temp_output,
                    parameters=mock_params,
                    max_requests_per_min=100.0,
                    num_workers=1,
                )
                mock_api.assert_called_once()
                result_index = 0
                with open(temp_output, encoding="utf-8") as file:
                    for line in file:
                        result = json.loads(line)
                        self.assertEqual(result["sample"]["id"], result_index)
                        result_index += 1
            self.assertEqual(result_index, len(samples))

    # Mock eval's handle of time.monotonic(), not the original. Otherwise, we'd
    # interfere with asyncio's sleeping and timeout functionality.
    @patch("eval.processing.monotonic", return_value=123456.0)
    async def test_rate_limit(self, mock_time: MagicMock, _: MagicMock) -> None:
        """Test that the request rate limit is enforced"""
        mock_params = create_mock_params()
        mock_prompt = "This is a test"
        max_requests_per_min = 1.0
        request_interval_s = 1.0 / (max_requests_per_min / 60.0)
        timeout_s = 0.1
        requests_queue: asyncio.Queue[Request[LawParameters]] = asyncio.Queue()
        task = asyncio.create_task(
            process_samples(
                samples=[LAW_SAMPLE] * 2,
                prompt_func=lambda s: mock_prompt,
                parameters=mock_params,
                requests_queue=requests_queue,
                max_requests_per_min=max_requests_per_min,
            )
        )
        # First request should be sent immediately
        await asyncio.wait_for(requests_queue.get(), timeout_s)
        # Second request should be delayed
        # Wrap it in a task so we can await it more than once.
        second_request = asyncio.create_task(requests_queue.get())
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(second_request),
                timeout_s,
            )
        # Advance time and try again
        mock_time.return_value += request_interval_s
        await asyncio.wait_for(second_request, timeout_s)
        # Wait for task to finish
        await asyncio.wait_for(task, timeout_s)


class TestFindMostRecentSample(unittest.TestCase):
    @contextmanager
    def write_results_file(
        self,
        contents: Iterable[Union[str, Mapping[str, Any]]],
    ) -> Iterator[Path]:
        """Write some test results to a temporary file and yield its path"""
        with make_temp_file() as temp_path:
            with open(temp_path, "w", encoding="utf-8") as temp_file:
                for line in contents:
                    if isinstance(line, str):
                        temp_file.write(line)
                    else:
                        json.dump(line, temp_file, indent=None)
                    temp_file.write("\n")
            # Close the file before yielding it for Windows compatibility
            yield temp_path

    def test_nonexistent_file(self) -> None:
        """Test that a nonexistent file returns None"""
        self.assertIsNone(find_most_recent_sample(Path("nonexistent.jsonl")))

    def test_empty_file(self) -> None:
        """Test that an empty file returns None"""
        with self.write_results_file([]) as temp_path:
            self.assertIsNone(find_most_recent_sample(temp_path))

    def test_single_sample(self) -> None:
        """Test that a file with one sample returns its id"""
        sample_id = 123
        with self.write_results_file([{"sample": {"id": sample_id}}]) as temp_path:
            self.assertEqual(find_most_recent_sample(temp_path), sample_id)

    def test_multiple_samples(self) -> None:
        """Test that the last id is returned, not the highest"""
        sample_ids = [3, 2, 1]
        with self.write_results_file(
            [{"sample": {"id": sample_id}} for sample_id in sample_ids]
        ) as temp_path:
            self.assertEqual(find_most_recent_sample(temp_path), sample_ids[-1])

    def test_invalid_samples(self) -> None:
        """Test that invalid samples are ignored"""
        sample_id = 321
        with self.write_results_file(
            [
                "{",
                {},
                {"sample": {"id": sample_id}},
                {"sample": None},
                {"sample": {"id": None}},
                {"sample": {"id": "abc"}},
                "}",
            ]
        ) as temp_path:
            self.assertEqual(find_most_recent_sample(temp_path), sample_id)


if __name__ == "__main__":
    unittest.main()
