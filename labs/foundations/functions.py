"""Pure functions are easy to test and compose."""
from collections.abc import Iterable

def batches(values: Iterable[int], size: int) -> list[list[int]]:
    if type(size) is not int or size <= 0:
        raise ValueError("positive integer size required")
    result, current = [], []
    for value in values:
        current.append(value)
        if len(current) == size:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result

def token_cost(input_tokens: int, output_tokens: int, *, input_rate: float, output_rate: float) -> float:
    if min(input_tokens, output_tokens, input_rate, output_rate) < 0:
        raise ValueError("negative quantity")
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

if __name__ == "__main__":
    print(batches(range(7), 3))
    print("hypothetical cost:", token_cost(1000, 200, input_rate=2, output_rate=8))
