"""A pipe represented as composable iterators."""
from collections import Counter
from typing import Iterable, Iterator

SAMPLE = ["INFO request=a tokens=4", "ERROR request=b timeout", "INFO request=c tokens=7",
          "ERROR request=d timeout", "ERROR request=e invalid_json"]

def select(lines: Iterable[str], prefix: str) -> Iterator[str]:
    for line in lines:
        if line.startswith(prefix):
            yield line

def error_counts(lines: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(line.split()[-1] for line in select(lines, "ERROR ")).items()))

if __name__ == "__main__":
    print(error_counts(SAMPLE))
