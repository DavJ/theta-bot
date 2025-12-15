#!/usr/bin/env python
import argparse
import json
from theta_bot_averaging.validation import run_walkforward


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    result = run_walkforward(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
