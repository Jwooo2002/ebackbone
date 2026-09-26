import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ebackbone_v3.seact_data import DTYPE, SeActDataset, prepare_inputs, collate, sha256


def events():
    a=np.zeros(4,dtype=DTYPE)
    a['x']=[0,1,1,345];a['y']=[0,1,1,259];a['t']=[100,125,175,200];a['p']=[0,1,1,1]
    return a


def fixture(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();man=tmp_path/'man';man.mkdir();cache=tmp_path/'cache';cache.mkdir()
    hashes={}
    for i,role in enumerate(('train','validation','test')):
        path=raw/(role+'.aedat4');path.write_bytes(('raw'+role).encode())
        a=events();p=cache/(role+'.npy');np.save(p,a,allow_pickle=False)
        metadata=dict(raw_sha256=sha256(path),cache_sha256=sha256(p),event_count=4,temporal_start=100,
                      temporal_end=200,decoder_version='seact-aedat4-lossless-1')
        p.with_suffix('.json').write_text(json.dumps(metadata))
        row=dict(sample_id=role,relative_path=path.name,class_label=i,split=role,
                 source_split='released_test' if role=='test' else 'released_train',raw_sha256=sha256(path),
                 raw_bytes=path.stat().st_size,raw_mtime_ns=path.stat().st_mtime_ns,cache_path=str(p),
                 cache_sha256=sha256(p),event_count=4,temporal_start=100,temporal_end=200)
        f=man/(role+'.jsonl');f.write_text(json.dumps(row)+'\n');hashes[role]=sha256(f)
    provenance=dict(version='seact-released-membership-v1',dataset='SeACT',dataset_root=str(raw),
                    source_height=260,source_width=346,height=288,width=352,num_classes=58,
                    manifest_sha256=hashes,split_counts=dict(train=1,validation=1,test=1),
                    decoder_version='seact-aedat4-lossless-1')
    (man/'provenance.json').write_text(json.dumps(provenance));return man,raw


def test_full_event_mass_alignment_native_padding_and_zero_duration():
    a=events();x=prepare_inputs(a)
    assert x['points'].shape==(4,4) and x['voxel_grid'].shape==(2,8,288,352)
    torch.testing.assert_close(x['points'][-1],torch.tensor([1.,1.,1.,1.]))
    assert x['event_frame'].expm1().sum().item()==pytest.approx(4.)
    assert x['voxel_grid'].expm1().sum().item()==pytest.approx(4.)
    assert torch.count_nonzero(x['event_frame'][:,260:])==0
    assert torch.count_nonzero(x['voxel_grid'][:,:,: ,346:])==0
    assert x['time_surface'][1,259,345]==1
    for mode in ('hierarchy_only','latent_only'):
        single=prepare_inputs(a,mode=mode)
        for key,value in single.items():assert torch.equal(value,x[key])
    a['t']=100;x=prepare_inputs(a)
    assert torch.all(x['points'][:,2]==1) and torch.all(x['voxel_lower']//(72*88)==7)
    a['x'][0]=346
    with pytest.raises(ValueError,match='invalid raw'):prepare_inputs(a)


def test_dataset_gates_identity_and_packed_sample_isolation(tmp_path):
    man,raw=fixture(tmp_path)
    with pytest.raises(ValueError,match='final test'):SeActDataset('/missing','/missing','test')
    with pytest.raises(ValueError,match='only to test'):SeActDataset(man,raw,'train',allow_final_test=True)
    d=SeActDataset(man,raw,'train');s=d[0];batch=collate([s,s])
    assert batch['inputs']['event_counts'].tolist()==[4,4]
    torch.testing.assert_close(batch['inputs']['voxel_lower'][4:],s['inputs']['voxel_lower']+8*72*88)
    assert s['source']['event_count']==4
    # Labels enter the classifier path only; changing label does not change rendering.
    d.rows[0].class_label=57
    for k,v in s['inputs'].items():assert torch.equal(v,d[0]['inputs'][k])
    (raw/'train.aedat4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw source changed'):d[0]


def test_manifest_and_cache_corruption_fail_closed(tmp_path):
    man,raw=fixture(tmp_path);d=SeActDataset(man,raw,'validation')
    cache=Path(d.rows[0].cache_path)
    cache.write_bytes(cache.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='cache identity'):d[0]
    (man/'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='manifest hash'):SeActDataset(man,raw,'train')
