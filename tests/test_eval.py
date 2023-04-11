import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Union

from eval import find_most_recent_sample
from tests.utils import make_temp_file


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
