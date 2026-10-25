"""Explicit freeze, publication, public-data capture and prospective model replay."""
import argparse
import json
from pathlib import Path

import requests

from theta_bot_averaging import entry_forward as forward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    freeze=commands.add_parser('freeze')
    freeze.add_argument('--protocol',type=Path,required=True)
    freeze.add_argument('--start')
    anchor=commands.add_parser('anchor')
    anchor.add_argument('--protocol',type=Path,required=True)
    anchor.add_argument('--out',type=Path,required=True)
    anchor.add_argument('--run-id',type=int)
    anchor.add_argument('--run-json',type=Path)
    anchor.add_argument('--tree-json',type=Path)
    capture=commands.add_parser('collect')
    capture.add_argument('--protocol',type=Path,required=True)
    capture.add_argument('--out',type=Path,default=Path('data/raw/entry_forward'))
    capture.add_argument('--raw',type=Path,help='Full three-symbol raw arrays from the connected Binance spot tool')
    capture.add_argument('--observed-at',help='Required for imported arrays; fixed completed-bar boundary')
    evaluate=commands.add_parser('evaluate')
    evaluate.add_argument('--protocol',type=Path,required=True)
    evaluate.add_argument('--anchor',type=Path,required=True)
    evaluate.add_argument('--data',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,default=Path('artifacts/entry_forward/accounts'))
    args=parser.parse_args()
    if args.action=='freeze':
        result=forward.freeze_protocol(args.protocol,args.start)
    elif args.action=='anchor':
        if args.run_json and args.tree_json and args.run_id is None:
            run=json.loads(args.run_json.read_text())
            tree=json.loads(args.tree_json.read_text())
        elif args.run_id and not args.run_json and not args.tree_json:
            url=f'https://api.github.com/repos/{forward.REPOSITORY}'
            response=requests.get(f'{url}/actions/runs/{args.run_id}',timeout=25)
            response.raise_for_status()
            run=response.json()
            response=requests.get(f"{url}/git/trees/{run['head_commit']['tree_id']}",params={'recursive':1},timeout=25)
            response.raise_for_status()
            tree=response.json()
        else:
            parser.error('Supply either --run-id or both --run-json and --tree-json')
        result=forward.write_publication(args.protocol,run,tree,args.out)
    elif args.action=='collect':
        if args.raw:
            if not args.observed_at:
                parser.error('--observed-at is required for imported raw arrays')
            result=str(forward.import_snapshot(args.protocol,json.loads(args.raw.read_text()),args.out,
                args.observed_at,'binance_mcp_public_spot'))
        else:
            if args.observed_at:
                parser.error('--observed-at is only accepted with --raw')
            result=str(forward.collect(args.protocol,args.out))
    else:
        result=forward.evaluate_forward(args.protocol,args.anchor,args.data,args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
