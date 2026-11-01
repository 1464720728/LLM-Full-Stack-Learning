"""Step 5: adapt a reusable analysis function to command-line input."""
import argparse
import json
from pathlib import Path
from .step04_pipeline import SAMPLE, analyze

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            report = analyze(SAMPLE)
        else:
            with args.input.open(encoding='utf-8') as stream:
                report = analyze(stream)
    except (OSError, UnicodeError) as error:
        parser.exit(2, f'input error: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
