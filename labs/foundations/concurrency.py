"""Bounded async workers; deterministic results, not timing claims."""
import asyncio
from dataclasses import dataclass

@dataclass
class Meter:
    active: int = 0
    peak: int = 0

async def squares(values, concurrency=2):
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("positive concurrency required")
    semaphore, meter = asyncio.Semaphore(concurrency), Meter()
    async def one(index, value):
        async with semaphore:
            meter.active += 1
            meter.peak = max(meter.peak, meter.active)
            try:
                await asyncio.sleep(0)  # yield, without a network or timing dependency
                return index, value * value
            finally:
                meter.active -= 1
    pairs = await asyncio.gather(*(one(i, v) for i, v in enumerate(values)))
    return [v for _, v in sorted(pairs)], meter

if __name__ == "__main__":
    result, meter = asyncio.run(squares(range(6), 2))
    print(result)
    print("peak=", meter.peak, "active_after=", meter.active)
