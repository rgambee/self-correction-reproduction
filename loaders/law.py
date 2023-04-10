import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator

from loaders import DatasetLoader, Question


@dataclass
class LawParameters:  # pylint: disable=too-many-instance-attributes
    """Parameters for a single question from the Law dataset

    The fields are as follows:
    * race: Amerindian, Asian, Black, Hispanic, Mexican, Other, Puertorican, or White
    * sex: encoded as 1 or 2 for an unknown reason
    * LSAT: score on the Law School Admission Test
    * UGPA: undergraduate GPA
    * region_first: FW, GL, MS, MW, Mt, NE, NG, NW, PO, SC, or SE
    * ZFYA: average grade during first year of law school
    * sander_index: Unknown
    * first_pf: Unknown for certain but possibly pass/fail during first year
    """

    # Pylint doesn't like capitalized field names, but we use them to match the columns
    # in the CSV file.
    # pylint: disable=invalid-name
    race: str
    sex: str
    LSAT: float
    UGPA: float
    region_first: str
    ZFYA: float
    sander_index: float
    first_pf: float

    def __post_init__(self) -> None:
        self.LSAT = float(self.LSAT)
        self.UGPA = float(self.UGPA)
        self.ZFYA = float(self.ZFYA)
        self.sander_index = float(self.sander_index)
        self.first_pf = float(self.first_pf)


class LawQuestion(Question[LawParameters]):
    pass


class LawLoader(DatasetLoader[LawParameters]):
    """Loader for the law school dataset

    The law school dataset is a CSV file with columns matching the fields in
    LawParameters, plus a leading unlabeled column for the student ID.
    """

    dataset = "law"

    def _entry_to_question(self, entry: Dict[str, Any]) -> LawQuestion:
        """Transform a line from the law school dataset into a Question"""
        param_dict = dict(entry)
        param_dict.pop("id")
        parameters = LawParameters(**param_dict)
        return LawQuestion(
            dataset=self.dataset,
            category="",
            id=int(entry["id"]),
            parameters=parameters,
            answers=["no", "yes"],
            correct_answer=int(parameters.first_pf),
        )

    def _iter_entries(self, path: Path) -> Iterator[LawQuestion]:
        """Loop over the lines of a CSV file and yield each as a question"""
        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                fieldnames=("id",) + tuple(fld.name for fld in fields(LawParameters)),
            )
            for i, entry in enumerate(reader):
                if i == 0 and entry["id"] == "":
                    # Skip header
                    continue
                yield self._entry_to_question(entry)
