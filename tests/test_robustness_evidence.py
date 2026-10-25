"""Independent standard-library audit of the published calendar/CSV evidence."""
import csv
from decimal import Decimal
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT=Path(__file__).resolve().parents[1]
STEM='THETABOT_ROBUSTNESS_2026-10-25'


def test_published_calendar_economics_recompute_from_raw_csv_with_decimal():
    folder=ROOT/'docs/evaluation'
    manifest=json.loads((folder/(STEM+'_EVIDENCE.zip.manifest.json')).read_text())
    chunks=[]
    for part in manifest['parts']:
        payload=(folder/part['file']).read_bytes()
        assert len(payload)==part['bytes']
        assert hashlib.sha256(payload).hexdigest()==part['sha256']
        chunks.append(payload)
    payload=b''.join(chunks)
    assert len(payload)==manifest['bytes'] and hashlib.sha256(payload).hexdigest()==manifest['sha256']
    report=json.loads(gzip.decompress((folder/(STEM+'.json.gz')).read_bytes()))
    proof=report['frozen_protocol']
    body={k:v for k,v in proof.items() if k!='protocol_sha256'}
    canonical=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==proof['protocol_sha256']
    assert report['live_eligible'] is False and report['reconciled_paths']==96
    assert report['preceding_control_checks']==24 and len(report['audited_accounts'])==14
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in proof['source_sha256'].items())
    checked_months=checked_windows=0
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert z.testzip() is None
        assert json.loads(z.read('summary.json'))==report
        hashes=json.loads(z.read('csv_hashes.json'))
        assert len(hashes)==64
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in hashes.items())
        for stem,account in report['audited_accounts'].items():
            endpoints={}
            for row in csv.DictReader(io.StringIO(z.read(stem+'_equity.csv').decode())):
                endpoints[row['timestamp'][:7]]=Decimal(row['equity'])
            assert list(endpoints)==[r['month'] for r in account['calendar']['months']]
            assert len(endpoints)==57
            starts=[Decimal(10000),*list(endpoints.values())[:-1]]
            returns=[]
            for (month,nav),start,stated in zip(endpoints.items(),starts,account['calendar']['months']):
                value=nav/start-1
                assert abs(value-Decimal(str(stated['return'])))<Decimal('1e-12')
                returns.append(value)
                checked_months+=1
            for year in account['calendar']['calendar_years']:
                locations=[i for i,month in enumerate(endpoints) if month.startswith(str(year['year']))]
                first,last=locations[0],locations[-1]
                value=list(endpoints.values())[last]/starts[first]-1
                assert abs(value-Decimal(str(year['return'])))<Decimal('1e-12')
                assert year['months']==len(locations)
                assert year['partial_year']==(len(locations)!=12)
            for width in (12,24):
                stated=account['calendar']['rolling_months'][str(width)]
                expected=[]
                for end in range(width,58):
                    value=list(endpoints.values())[end-1]/starts[end-width]-1
                    assert abs(value-Decimal(str(stated['windows'][end-width]['return'])))<Decimal('1e-12')
                    expected.append(value)
                    checked_windows+=1
                assert stated['count']==len(expected)
                assert stated['negative_count']==sum(v<0 for v in expected)
                assert abs(min(expected)-Decimal(str(stated['worst_return'])))<Decimal('1e-12')
            retained=sorted(returns,reverse=True)[3:]
            product=Decimal(1)
            for value in retained:
                product*=1+value
            assert abs(product-1-Decimal(str(account['calendar']['mathematical_return_excluding_best_three_months'])))<Decimal('1e-12')
            ending=list(endpoints.values())[-1]
            attributed=sum(Decimal(str(a['net_pnl_contribution'])) for a in account['attribution']['assets'].values())
            assert abs(attributed-(ending-10000))<Decimal('1e-7')
    assert checked_months==798 and checked_windows==1120
