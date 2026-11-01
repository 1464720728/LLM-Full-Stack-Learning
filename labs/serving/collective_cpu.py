"""Bounded CPU process-group lesson; coordinator sum, not NCCL or a benchmark."""
import argparse
import json
import multiprocessing as mp
from queue import Empty


def _worker(rank, values, incoming, outgoing, timeout):
    incoming.put(("contribution", rank, values))
    result = outgoing.get(timeout=timeout)
    incoming.put(("result", rank, result))


def coordinator_allreduce(contributions, timeout=5.0):
    """Return each rank's sum while checking identities and bounded cleanup."""
    if not 1 <= len(contributions) <= 4:
        raise ValueError("this lesson allows 1..4 ranks")
    width = len(contributions[0])
    if width == 0 or any(len(row) != width for row in contributions):
        raise ValueError("all ranks must contribute the same nonempty shape")
    if any(type(value) is not int for row in contributions for value in row):
        raise ValueError("integer contributions required for exact comparison")
    if not 0 < timeout <= 10:
        raise ValueError("timeout must be in (0, 10] seconds")
    context = mp.get_context("spawn")
    incoming = context.Queue()
    outgoing = [context.Queue() for _ in contributions]
    processes = [context.Process(target=_worker, args=(rank, list(values),
                 incoming, outgoing[rank], timeout))
                 for rank, values in enumerate(contributions)]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        received = {}
        for _ in contributions:
            kind, rank, values = incoming.get(timeout=timeout)
            if kind != "contribution" or rank in received or not 0 <= rank < len(outgoing):
                raise RuntimeError("invalid contribution identity")
            if len(values) != width:
                raise RuntimeError("incompatible contribution shape")
            received[rank] = values
        total = [sum(received[rank][column] for rank in range(len(outgoing)))
                 for column in range(width)]
        for channel in outgoing:
            channel.put(total)
        results = {}
        for _ in contributions:
            kind, rank, values = incoming.get(timeout=timeout)
            if kind != "result" or rank in results or not 0 <= rank < len(outgoing):
                raise RuntimeError("invalid result identity")
            results[rank] = values
        for process in started:
            process.join(timeout)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("rank did not finish successfully")
        if any(value != total for value in results.values()):
            raise AssertionError("ranks disagree on the reduced result")
        return {"kind": "CPU_coordinator_semantics_not_NCCL_performance",
                "ranks": len(contributions), "sum": total,
                "results": [results[rank] for rank in range(len(outgoing))]}
    except Empty as error:
        raise RuntimeError("bounded collective timed out") from error
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout)
        incoming.close()
        for channel in outgoing:
            channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    contributions = [[rank+1, 10*(rank+1)] for rank in range(args.ranks)]
    print(json.dumps(coordinator_allreduce(contributions), indent=2))


if __name__ == "__main__":
    main()
