"""A CLI parser is a translation boundary between strings and domain types."""
import argparse
import json
from .functions import batches

def positive(text):
    try:
        value = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError("integer required") from e
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--count", type=positive, default=7)
    result.add_argument("--batch-size", type=positive, default=3)
    return result

def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(batches(range(args.count), args.batch_size)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
