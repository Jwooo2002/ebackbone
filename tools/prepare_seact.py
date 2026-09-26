"""CPU-only, lossless AEDAT4 cache preparation; no held-out event decoding."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import numpy as np

DECODER_VERSION = 'seact-aedat4-lossless-1'
DTYPE = np.dtype([('t', '<i8'), ('x', '<i2'), ('y', '<i2'), ('p', 'i1')])

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def decode(raw, output, expected_sha=None):
    from dv import AedatFile
    import importlib.metadata
    actual = digest(raw)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError('raw source digest changed')
    meta_path = output.with_suffix('.json')
    if output.exists():
        meta = json.loads(meta_path.read_text())
        if meta['raw_sha256'] != actual or meta['decoder_version'] != DECODER_VERSION or meta['cache_sha256'] != digest(output):
            raise ValueError('cache identity mismatch')
        return meta
    arrays = []
    with AedatFile(str(raw)) as source:
        if tuple(source['events'].size) != (260, 346):
            raise ValueError('unexpected sensor geometry')
        for block in source['events'].numpy():
            a = np.empty(len(block), dtype=DTYPE)
            for key, original in [('t','timestamp'),('x','x'),('y','y'),('p','polarity')]:
                a[key] = block[original]
            arrays.append(a)
    a = np.concatenate(arrays)
    if not len(a) or np.any(a['t'][1:] < a['t'][:-1]) or np.any((a['x']<0)|(a['x']>=346)|(a['y']<0)|(a['y']>=260)) or not set(np.unique(a['p'])) <= {0,1}:
        raise ValueError('invalid raw events; no filtering permitted')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npy')
    np.save(temporary,a,allow_pickle=False)
    meta = dict(raw_sha256=actual, cache_sha256=digest(temporary), event_count=len(a),
                temporal_start=int(a['t'][0]),temporal_end=int(a['t'][-1]),
                x_range=[int(a['x'].min()),int(a['x'].max())], y_range=[int(a['y'].min()),int(a['y'].max())],
                polarity_values=[int(v) for v in np.unique(a['p'])], decoder_version=DECODER_VERSION,
                dv_version=importlib.metadata.version('dv'),numpy_version=np.__version__,
                source_height=260,source_width=346,events_dropped=0,events_sampled=0,
                raw_bytes=raw.stat().st_size,raw_mtime_ns=raw.stat().st_mtime_ns)
    os.replace(temporary,output)
    write_json(meta_path,meta)
    return meta

def key(path):
    path=Path(str(path).strip())
    match=re.search(r'(\d{4}-\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.aedat4$',path.name)
    if not match: raise ValueError(f'unrecognized recording {path}')
    return path.parent.name+'/'+match[1]+'.aedat4'

def prepare(root,released,out):
    manifests=out/'manifests'/'released-v1'
    if (manifests/'provenance.json').exists():
        raise FileExistsError('immutable manifest already completed')
    available={}
    for p in root.rglob('*.aedat4'):
        k=key(p)
        if k in available: raise ValueError('duplicate canonical raw identity')
        available[k]=p
    mapping=json.loads((released/'SeAct_idx_to_label.json').read_text())
    memberships={role:[key(line) for line in (released/name).read_text().splitlines() if line.strip()]
                 for role,name in [('released_train','SeAct_train.txt'),('released_test','SeAct_val.txt')]}
    if len(memberships['released_train']) !=464 or len(memberships['released_test'])!=116 or set(memberships['released_train'])&set(memberships['released_test']) or len(set(sum(memberships.values(),[])))!=580 or set(available)!=set(sum(memberships.values(),[])):
        raise ValueError('released membership mismatch')
    labels={k:int(mapping[str(int(Path(k).name[:4]))]) for k in available}
    validation=set()
    for label in range(58):
        choices=[k for k in memberships['released_train'] if labels[k]==label]
        if len(choices)<2: raise ValueError('class lacks training support')
        validation.add(min(choices,key=lambda k:hashlib.sha256(('seact-internal-val-20260908\0'+k).encode()).digest()))
    rows={s:[] for s in ('train','validation','test')}
    for i,k in enumerate(sorted(available)):
        p=available[k]
        role='released_train' if k in memberships['released_train'] else 'released_test'
        split=('validation' if k in validation else 'train') if role=='released_train' else 'test'
        raw_hash=digest(p)
        cache=out/'raw_cache'/(raw_hash+'.npy')
        meta=decode(p,cache,raw_hash) if split!='test' else {}
        row=dict(sample_id=k,relative_path=str(p.relative_to(root)),class_label=labels[k],split=split,
                 source_split=role,raw_sha256=raw_hash,raw_bytes=p.stat().st_size,raw_mtime_ns=p.stat().st_mtime_ns,
                 cache_path=str(cache.resolve()),event_count=meta.get('event_count'),
                 temporal_start=meta.get('temporal_start'),temporal_end=meta.get('temporal_end'),
                 cache_sha256=meta.get('cache_sha256'))
        rows[split].append(row)
        print(json.dumps(dict(completed=i+1,total=580,split=split,sample_id=k,event_count=row['event_count'])),flush=True)
    manifests.mkdir(parents=True,exist_ok=True)
    for split,items in rows.items():
        target=manifests/(split+'.jsonl')
        if target.exists():raise FileExistsError(target)
        target.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in items))
    provenance=dict(version='seact-released-membership-v1',dataset='SeACT',dataset_root=str(root.resolve()),
                    source_height=260,source_width=346,height=288,width=352,num_classes=58,
                    split_counts={s:len(r) for s,r in rows.items()},
                    manifest_sha256={s:digest(manifests/(s+'.jsonl')) for s in rows},
                    released_sources={name:dict(path=str((released/name).resolve()),sha256=digest(released/name)) for name in ('SeAct_train.txt','SeAct_val.txt','SeAct_idx_to_label.json')},
                    validation_policy='one sample per class from released train; minimum SHA256 of seact-internal-val-20260908 NUL sample_id',
                    final_test_policy='released SeAct_val membership; no decode before selected best; 53 of 58 classes represented',
                    split_classes={s:sorted(set(r['class_label'] for r in items)) for s,items in rows.items()},
                    test_events_decoded=False,decoder_version=DECODER_VERSION,
                    decoder_python=sys.executable,decoder_script=str(Path(__file__).resolve()),decoder_script_sha256=digest(__file__),
                    local_name_mapping='subject plus 4-digit code and complete timestamp; local khao/xu prefix omitted')
    write_json(manifests/'provenance.json',provenance)
    print(json.dumps(dict(status='complete',manifest_dir=str(manifests),split_counts=provenance['split_counts'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('decode');d.add_argument('--raw',type=Path,required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--raw-sha256',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,required=True);a.add_argument('--released',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='decode':print(json.dumps(decode(args.raw,args.output,args.raw_sha256)))
    else:prepare(args.root,args.released,args.output)
