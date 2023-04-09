from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).parent
BBQ_DATASET_DIR = CURRENT_DIR / "BBQ" / "data"
LAW_DATASET_DIR = CURRENT_DIR
WINOGENDER_DATASET_DIR = CURRENT_DIR / "datasets" / "winogender-schemas"


def find_bbq_dataset(search_dir: Path = BBQ_DATASET_DIR) -> Iterable[Path]:
    return search_dir.glob("*.jsonl")


def find_law_dataset(search_dir: Path = LAW_DATASET_DIR) -> Iterable[Path]:
    return [search_dir / "law_data.csv"]


def find_winogender_dataset(
    search_dir: Path = WINOGENDER_DATASET_DIR,
) -> Iterable[Path]:
    return [search_dir / "all_sentences.tsv"]


def find_winogender_stats(search_dir: Path = WINOGENDER_DATASET_DIR) -> Path:
    return search_dir / "occupation-stats.tsv"
