import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional


@contextmanager
def make_temp_file() -> Iterator[Path]:
    """Create a temporary file for use in tests"""
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="wt", delete=False) as file:
            temp_path = Path(file.name)
        # Close the file before yielding it for Windows compatibility
        yield temp_path
    finally:
        if temp_path:
            temp_path.unlink()


@contextmanager
def write_dummy_dataset(entries: Iterable[str]) -> Iterator[Path]:
    """Write a dummy dataset to a tempfile and yield its path"""
    with make_temp_file() as temp_path:
        with open(temp_path, "w", encoding="utf-8") as file:
            file.writelines(entries)
        # Close the file before yielding it for Windows compatibility
        yield temp_path
