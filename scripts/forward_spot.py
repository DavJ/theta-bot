"""Freeze a prospective study or replay its append-only forward spot account."""
import argparse
import json
from pathlib import Path

from spot_bot.forward_validation import freeze_protocol, evaluate_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--protocol', type=Path, required=True)
    freeze.add_argument('--start', help='Future UTC midnight; default next UTC day')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--protocol', type=Path, required=True)
    evaluate.add_argument('--data', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, default=Path('artifacts/forward_spot/accounts'))
    args = parser.parse_args()
    result = (freeze_protocol(args.protocol, args.start) if args.action == 'freeze'
              else evaluate_forward(args.protocol, args.data, args.out))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
