"""Streaming batches retain at most one batch, not the whole dataset."""
from collections.abc import Iterable, Iterator
from typing import TypeVar
T = TypeVar("T")

def stream_batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    if type(size) is not int or size <= 0:
        raise ValueError("positive integer size required")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    source = (i * i for i in range(7))
    print(list(stream_batches(source, 3)))
    print("consumed source:", list(source))
