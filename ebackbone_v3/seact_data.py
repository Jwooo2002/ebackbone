"""Full-recording SeACT raw arrays, immutable split membership and aligned rendering."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from .dual_fusion_data import input_keys, collate as packed_collate

INPUT_CONTRACT = dict(version='seact-full-recording-native-padded-1', source_height=260,
    source_width=346, height=288, width=352, point_normalization='x/345,y/259,t_full_support,2p-1',
    voxel_bins=8, spatial_cell=4, tau_duration_ratio=.2, frame_normalization='log1p',
    voxel_normalization='log1p', surface_background=0, spatial_mapping='native coordinates; bottom/right zero padding',
    support='whole stored recording; closed endpoints', filtering='none', sampling='none', augmentation='none')
INPUT_CONTRACT_SHA256 = hashlib.sha256(json.dumps(INPUT_CONTRACT, sort_keys=True).encode()).hexdigest()
DTYPE = np.dtype([('t','<i8'),('x','<i2'),('y','<i2'),('p','i1')])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_inputs(events, *, mode='dual'):
    input_keys(mode)
    if events.dtype != DTYPE or events.ndim != 1 or not len(events):
        raise ValueError('require nonempty canonical raw structured events')
    x, y, t, p = (events[key] for key in ('x','y','t','p'))
    if (np.any(t[1:] < t[:-1]) or np.any((x<0)|(x>=346)|(y<0)|(y>=260))
            or np.any((p!=0)&(p!=1))):
        raise ValueError('invalid raw events; no filtering or truncation permitted')
    span = int(t[-1]) - int(t[0])
    normalized = (t.astype(np.float64) - int(t[0])) / span if span else np.ones(len(t), np.float64)
    position = normalized * 7
    lo = np.floor(position).astype(np.int64)
    hi = np.minimum(lo+1, 7)
    alpha = (position-lo).astype(np.float32)
    result = {'event_counts': torch.tensor([len(t)], dtype=torch.int64)}
    if mode != 'latent_only':
        spatial = (y.astype(np.int64)//4)*88 + x.astype(np.int64)//4
        result.update(points=torch.from_numpy(np.stack((x.astype(np.float32)/345, y.astype(np.float32)/259,
                      normalized.astype(np.float32), 2*p.astype(np.float32)-1),axis=1)),
                      voxel_lower=torch.from_numpy(lo*72*88+spatial), voxel_upper=torch.from_numpy(hi*72*88+spatial),
                      alpha=torch.from_numpy(alpha))
    if mode != 'hierarchy_only':
        ix, iy, ip = (a.astype(np.int64) for a in (x,y,p))
        frame = np.zeros((2,288,352),np.float32)
        np.add.at(frame,(ip,iy,ix),1)
        voxel = np.zeros((2,8,288,352),np.float32)
        np.add.at(voxel,(ip,lo,iy,ix),1-alpha)
        np.add.at(voxel,(ip,hi,iy,ix),alpha)
        latest = np.full_like(frame,-np.inf)
        np.maximum.at(latest,(ip,iy,ix),normalized.astype(np.float32))
        surface = np.zeros_like(frame)
        valid = np.isfinite(latest)
        surface[valid] = np.exp(-(1-latest[valid])/np.float32(.2))
        result.update(event_frame=torch.from_numpy(np.log1p(frame)),voxel_grid=torch.from_numpy(np.log1p(voxel)),
                      time_surface=torch.from_numpy(surface))
    return result


class SeActDataset(Dataset):
    def __init__(self, manifest_dir, dataset_root, split, *, mode='dual', allow_final_test=False):
        if split not in ('train','validation','test') or (split=='test' and not allow_final_test):
            raise ValueError('final test requires explicit allow_final_test=True')
        if split!='test' and allow_final_test:
            raise ValueError('allow_final_test applies only to test')
        input_keys(mode)
        self.root = Path(dataset_root).resolve()
        self.manifest_dir = Path(manifest_dir).resolve()
        self.provenance = json.loads((self.manifest_dir/'provenance.json').read_text())
        provenance = self.provenance
        if (provenance['version']!='seact-released-membership-v1' or provenance['dataset']!='SeACT'
                or Path(provenance['dataset_root']).resolve()!=self.root
                or [provenance[k] for k in ('source_height','source_width','height','width','num_classes')]!=[260,346,288,352,58]):
            raise ValueError('dataset geometry or provenance mismatch')
        all_rows = {}
        ids, raw_hashes = set(), set()
        for role in ('train','validation','test'):
            path = self.manifest_dir/(role+'.jsonl')
            if sha256(path)!=provenance['manifest_sha256'][role]:
                raise ValueError('immutable manifest hash mismatch')
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            if len(rows)!=provenance['split_counts'][role]:
                raise ValueError('split size mismatch')
            for row in rows:
                relative = Path(row['relative_path'])
                if relative.is_absolute() or '..' in relative.parts or row['split']!=role or not 0<=row['class_label']<58:
                    raise ValueError('invalid sample identity, label, or split')
                if row['source_split']!=('released_test' if role=='test' else 'released_train'):
                    raise ValueError('released split membership mismatch')
                if row['sample_id'] in ids or row['raw_sha256'] in raw_hashes:
                    raise ValueError('recording overlap or duplicate across splits')
                ids.add(row['sample_id']); raw_hashes.add(row['raw_sha256'])
            all_rows[role]=rows
        self.rows=tuple(SimpleNamespace(**r) for r in all_rows[split])
        self.split,self.mode=split,mode
        self.manifest_sha256=provenance['manifest_sha256'][split]
        self._verified_raw=set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        started=time.perf_counter()
        raw=self.root/row.relative_path
        st=raw.stat()
        if st.st_size!=row.raw_bytes or st.st_mtime_ns!=row.raw_mtime_ns:
            raise ValueError('raw source changed')
        if row.sample_id not in self._verified_raw:
            if sha256(raw)!=row.raw_sha256:
                raise ValueError('raw content hash mismatch')
            self._verified_raw.add(row.sample_id)
        cache=Path(row.cache_path)
        if not cache.exists():
            if self.split!='test':
                raise FileNotFoundError('prepared training cache missing')
            decoder=Path(self.provenance['decoder_script'])
            if sha256(decoder)!=self.provenance['decoder_script_sha256']:
                raise ValueError('AEDAT decoder source changed')
            subprocess.run([self.provenance['decoder_python'],str(decoder),'decode','--raw',str(raw),
                            '--output',str(cache),'--raw-sha256',row.raw_sha256],check=True,capture_output=True,text=True)
        metadata=json.loads(cache.with_suffix('.json').read_text())
        actual_hash=sha256(cache)
        if (metadata['raw_sha256']!=row.raw_sha256 or metadata['decoder_version']!=self.provenance['decoder_version']
                or metadata['cache_sha256']!=actual_hash or (row.cache_sha256 is not None and row.cache_sha256!=actual_hash)):
            raise ValueError('raw cache identity/hash mismatch')
        events=np.load(cache,allow_pickle=False,mmap_mode='r')
        if events.dtype!=DTYPE or len(events)!=metadata['event_count'] or (row.event_count is not None and len(events)!=row.event_count):
            raise ValueError('cached event count or dtype mismatch')
        if int(events['t'][0])!=metadata['temporal_start'] or int(events['t'][-1])!=metadata['temporal_end']:
            raise ValueError('cached full temporal support mismatch')
        decoded=time.perf_counter()
        inputs=prepare_inputs(events,mode=self.mode)
        return dict(inputs=inputs,label=row.class_label,sample_id=row.sample_id,split=self.split,
                    source_split=row.source_split,decode_seconds=decoded-started,render_seconds=time.perf_counter()-decoded,
                    source=dict(sample_id=row.sample_id,event_count=len(events),temporal_start=int(events['t'][0]),
                                temporal_end=int(events['t'][-1]),raw_sha256=row.raw_sha256),
                    input_contract_sha256=INPUT_CONTRACT_SHA256)


def collate(samples,*,height=288,width=352):
    return packed_collate(samples,height=height,width=width)
