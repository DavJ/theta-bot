"""Publish an immutable dated capture without changing the frozen depth model."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
import zipfile

from scripts.research_paper_execution import (
    collect, exclusive, export, read_protocol, save_and_analyze, utc,
)

LEGACY_STEM = 'THETABOT_DEPTH_PAPER_2026-10-23'


def reproducible_zip(source):
    """Keep evidence bytes; normalize ZIP metadata so replay is idempotent."""
    buffer = BytesIO()
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(buffer, 'w') as target:
        names = original.namelist()
        if (len(names) != len(set(names)) or any(
                n.startswith('/') or '..' in PurePosixPath(n).parts for n in names)):
            raise ValueError('Ambiguous evidence archive paths')
        for name in sorted(names):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            target.writestr(info, original.read(name), compresslevel=9)
    return buffer.getvalue()


def publish_report(report, coverage, out, destination):
    key = report.get('capture_sha256')
    if not isinstance(key,str) or re.fullmatch('[0-9a-f]{64}',key) is None:
        raise ValueError('A full capture SHA-256 is required')
    observed = utc(report['observed_at'])
    stem = f"THETABOT_DEPTH_PAPER_{observed.strftime('%Y-%m-%dT%H%M%SZ')}_{key[:12]}"
    with TemporaryDirectory(prefix='depth-report-') as folder:
        folder = Path(folder)
        export(report,coverage,Path(out),folder)
        markdown = (folder/(LEGACY_STEM+'.md')).read_text().splitlines()
        markdown[0] = '# Public spot-depth paper calibration — '+observed.date().isoformat()
        text = '\n'.join(markdown)+'\n'
        text = text.replace('One batch does not estimate tail costs', 'This short sample does not estimate tail costs')
        text = text.split('## Reproduce / explicitly collect another batch')[0]
        text += ('## Continue this frozen study\n\n'
            'The table describes the latest capture. The ZIP retains all saved capture and analysis receipts.\n'
            'To resume published coverage, restore the verified evidence ZIP into the account output directory,\n'
            'naming frozen_protocol.json as protocol.json and preserving captures/ and reports/.\n'
            'Use the Python and requests versions recorded in the protocol.\n\n'
            '```bash\n'
            'python -m scripts.write_depth_paper_report \\\n'
            '  --protocol docs/evaluation/THETABOT_DEPTH_PROTOCOL_2026-10-23.json \\\n'
            '  --out artifacts/depth_paper --collect\n'
            '# Or replace --collect with --capture PATH_TO_MATCHING_CAPTURE.json.\n'
            '```\n\n'
            'Each invocation collects or replays one explicit batch. Published filenames include\n'
            'the UTC observation time and capture hash; conflicting files are never overwritten.\n')
        archive = reproducible_zip(folder/(LEGACY_STEM+'_EVIDENCE.zip'))
        archive_name = stem+'_EVIDENCE.zip'
        files = {stem+'.md':text.encode(),
            stem+'.json.gz':(folder/(LEGACY_STEM+'.json.gz')).read_bytes(),
            archive_name:archive,
            stem+'_EVIDENCE.sha256':(hashlib.sha256(archive).hexdigest()+'  '+archive_name+'\n').encode()}
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    # Check the whole set before touching any existing publication.
    for name,payload in files.items():
        path = destination/name
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f'Immutable published capture differs: {name}')
    for name,payload in files.items():
        path = destination/name
        try:
            with path.open('xb') as f:
                f.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f'Concurrent publication differs: {name}')
    return {name:str(destination/name) for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--collect',action='store_true',help='Explicitly collect one public batch')
    action.add_argument('--capture',type=Path,help='Replay a matching saved capture')
    parser.add_argument('--out',type=Path,default=Path('artifacts/depth_paper'))
    parser.add_argument('--report-out',type=Path,default=Path('docs/evaluation'))
    args = parser.parse_args()
    proof = read_protocol(args.protocol)
    capture = collect(proof) if args.collect else json.loads(args.capture.read_text())
    report,coverage = save_and_analyze(proof,capture,args.out)
    exclusive(args.out/'protocol.json',proof)
    files = publish_report(report,coverage,args.out,args.report_out)
    print(json.dumps({'capture_sha256':report['capture_sha256'],'published':files,
        'coverage':coverage,'observed_executable_fills':0,'strategy_pnl_available':False},indent=2))


if __name__=='__main__':
    main()
