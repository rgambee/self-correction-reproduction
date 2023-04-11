import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


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
