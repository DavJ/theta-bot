"""Publication must preserve older evidence and be repeatable across ZIP clocks."""
import hashlib
from io import BytesIO
import zipfile

import pytest

from scripts import write_depth_paper_report as writer


@pytest.fixture
def published(tmp_path,monkeypatch):
    clock=[(2026,10,23,0,0,0)]
    def fake_export(report,coverage,out,destination):
        (destination/(writer.LEGACY_STEM+'.md')).write_text('# Original\n\nEvidence\n')
        (destination/(writer.LEGACY_STEM+'.json.gz')).write_bytes(b'stable compressed report')
        with zipfile.ZipFile(destination/(writer.LEGACY_STEM+'_EVIDENCE.zip'),'w') as z:
            z.writestr(zipfile.ZipInfo('captures/sample.json',clock[0]),b'{"raw":1}')
            z.writestr(zipfile.ZipInfo('reports/sample.json',clock[0]),b'{"audited":true}')
    monkeypatch.setattr(writer,'export',fake_export)
    report={'capture_sha256':'a'*64,'observed_at':'2026-10-25T01:02:03.456Z'}
    return report,tmp_path,clock


def test_replay_is_idempotent_despite_source_zip_timestamp_changes(published):
    report,root,clock=published
    files=writer.publish_report(report,{},root,root/'published')
    first={name:open(path,'rb').read() for name,path in files.items()}
    assert all('2026-10-25T010203Z_aaaaaaaaaaaa' in n for n in files)
    clock[0]=(2030,1,1,0,0,0)
    again=writer.publish_report(report,{},root,root/'published')
    assert first=={name:open(path,'rb').read() for name,path in again.items()}
    archive_name=next(n for n in files if n.endswith('.zip'))
    checksum=next(v for n,v in first.items() if n.endswith('.sha256')).decode().split()
    assert checksum==[hashlib.sha256(first[archive_name]).hexdigest(),archive_name]
    with zipfile.ZipFile(BytesIO(first[archive_name])) as z:
        assert z.testzip() is None and z.read('captures/sample.json')==b'{"raw":1}'


def test_conflicting_publication_does_not_change_any_file(published):
    report,root,_=published
    files=writer.publish_report(report,{},root,root/'published')
    markdown=next(path for name,path in files.items() if name.endswith('.md'))
    with open(markdown,'wb') as f: f.write(b'user edit')
    before={name:open(path,'rb').read() for name,path in files.items()}
    with pytest.raises(FileExistsError,match='Immutable'):
        writer.publish_report(report,{},root,root/'published')
    assert before=={name:open(path,'rb').read() for name,path in files.items()}


def test_invalid_identity_cannot_publish_paths(published):
    report,root,_=published
    report['capture_sha256']='../../other-file'
    with pytest.raises(ValueError,match='SHA-256'):
        writer.publish_report(report,{},root,root/'published')
    assert not (root/'published').exists()
