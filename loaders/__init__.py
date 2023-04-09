from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

# P is meant to be a dataclass representing the parameters for a particular question.
# However, it's not possible to actually enforce this since there's no type for an
# arbitrary dataclass.
P = TypeVar("P")


@dataclass
class Question(Generic[P]):
    """A single question from a dataset"""

    dataset: str
    category: str
    # pylint doesn't like two letter names, claiming they don't conform to the
    # snake_case convention
    id: int  # pylint: disable=invalid-name
    parameters: P
    answers: Sequence[str]
    correct_answer: int

    def __post_init__(self) -> None:
        self.id = int(self.id)


class DatasetLoader(Generic[P], ABC):
    """Abstract base class for loading datasets"""

    dataset: str

    def __init__(self, paths: Union[Path, Iterable[Path]]) -> None:
        # If paths is a single Path, wrap it in a list so we can iterate over it.
        # mypy isn't thrilled about this, hence the various `type:` comments.
        try:
            iter(paths)  # type: ignore[arg-type]
        except TypeError:
            self.paths: Iterable[Path] = [cast(Path, paths)]
        else:
            self.paths = cast(Iterable[Path], paths)

    @abstractmethod
    def _entry_to_question(self, entry: Any) -> Optional[Question[P]]:
        """Transform a line from the dataset into a Question"""

    def _iter_entries(self, path: Path) -> Iterator[Question[P]]:
        """Loop over the lines of a file and yield each as a question"""
        with open(path, encoding="utf-8") as file:
            for entry in file:
                question = self._entry_to_question(entry)
                if question is None:
                    continue
                yield question

    def __iter__(self) -> Iterator[Question[P]]:
        """Loop over the dataset files and yield all the questions"""
        for path in self.paths:
            yield from self._iter_entries(path)
