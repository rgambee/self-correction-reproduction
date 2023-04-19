from enum import Enum, unique


@unique
class Assessment(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"
