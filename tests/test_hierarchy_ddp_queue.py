"""Study queue ordering tests; no subprocesses or GPUs are launched."""
import importlib.util
import json
from pathlib import Path
import pytest


def queue(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[2]/'ebackbone_v3_hierarchy_ddp_b64_e50_artifacts'
    spec=importlib.util.spec_from_file_location('queue_under_test',root/'supervise.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    plan=json.loads((root/'plan.json').read_text())
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    for f,value in [('cpu_verification.json',{'status':'PASS'}),('protected_before.json',{}),('completed_current_protected.json',{})]:
        (tmp_path/f).write_text(json.dumps(value))
    monkeypatch.setattr(q,'ROOT',tmp_path);monkeypatch.setattr(q,'STATUS',tmp_path/'queue_status.json')
    monkeypatch.setattr(q,'export_results',lambda p:None)
    events=[]
    monkeypatch.setattr(q,'wait_current',lambda p:events.append('current_finished'))
    monkeypatch.setattr(q,'run_child',lambda p,n,m:events.append((m,n)))
    return q,events,plan


def test_all_preflights_precede_any_training(tmp_path,monkeypatch):
    q,events,plan=queue(tmp_path,monkeypatch);q.main()
    names=[v['name'] for v in plan['variants']]
    assert events==['current_finished']+[('preflight',n) for n in names]+[('train',n) for n in names]
    assert json.loads(q.STATUS.read_text())['stage']=='complete'
    with pytest.raises(RuntimeError,match='Existing queue'):q.main()


def test_failed_preflight_prevents_all_training(tmp_path,monkeypatch):
    q,events,_=queue(tmp_path,monkeypatch)
    def fail(plan,name,mode):
        events.append((mode,name))
        raise RuntimeError('preflight failed')
    monkeypatch.setattr(q,'run_child',fail)
    with pytest.raises(RuntimeError,match='preflight failed'):q.main()
    assert events==['current_finished',('preflight','hierarchy')]
