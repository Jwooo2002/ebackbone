from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ebackbone_v3 import hierarchy_ddp as d
from tests.test_hierarchy import sample
from tests.test_hierarchy_polarity_training import assert_nested_equal


class TinyData:
    def __init__(self,split='train',size=5):
        self.split=split
        self.rows=[SimpleNamespace(sample_id=f'{split}/{i}') for i in range(size)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        s=sample(i,self.split)
        s['inputs']['points'][:,3]=2*(torch.arange(5+i)%2)-1
        s['inputs']['time_surface']=torch.rand(2,32,32,generator=torch.Generator().manual_seed(i))
        return s


class SelectData(TinyData):
    def __init__(self,name,**kwargs):super().__init__(**kwargs);self.name=name
    def __getitem__(self,i):
        s=super().__getitem__(i)
        if self.name not in d.TS_VARIANTS:del s['inputs']['time_surface']
        return s


def test_exact_shards_full_manifest_and_edge_tails():
    for size in [2,3,63,64,65,66,127,128,129,5000,124395]:
        for train in (True,False):
            shards=[d.ExactBatchShard(size,r,2,32,20260908,1,train) for r in range(2)]
            ids=[[i for b in s for i in b] for s in shards]
            assert set(ids[0]).isdisjoint(ids[1])
            assert sorted(ids[0]+ids[1])==list(range(size))
            assert all(0<len(b)<=32 for s in shards for b in s)
            if train:
                assert len(shards[0])==len(shards[1])
                for i,(a,b) in enumerate(zip(*[s.batches for s in shards])):
                    assert len(a)+len(b)==shards[0].global_sizes[i]
                again=d.ExactBatchShard(size,0,2,32,20260908,1,True)
                assert again.batches==shards[0].batches
    assert d.ExactBatchShard(124395,0,2,32,20260908,1,True).global_sizes[-1]==43


def worker(rank,init,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    device=torch.device('cpu')
    settings=dict(batch_size=2,epochs=50,accumulation_steps=1,seed=20260908,cpu_threads=1,num_workers=0,
                  learning_rate=.05,momentum=.9,weight_decay=.0001)
    reports={}
    for name in d.VARIANTS:
        d.seed_all(settings,device)
        m=d.make_model(name,32,32)
        ddp=DDP(m,broadcast_buffers=False)
        opt=torch.optim.SGD(m.parameters(),lr=.05,momentum=.9,weight_decay=.0001)
        data=SelectData(name,size=7)
        # Two rank averages plus the direct weighted-gradient reference in probe.
        tr=d.run_epoch(ddp,data,name,settings,device,1,opt,probe=True,height=32,width=32)
        val=d.run_epoch(ddp,SelectData(name,split='validation',size=5),name,settings,device,1,height=32,width=32)
        assert tr['samples']==7 and val['samples']==5 and val['rank_sample_counts']==[3,2]
        assert tr['duplicates']==val['duplicates']==0
        # One shared model is compared with the sum-loss global mean. This covers
        # unequal rank sizes, independently of the explicit all-reduce check.
        ref=d.make_model(name,32,32);ref.load_state_dict(m.state_dict())
        x=d.collator(name,32,32)([data[i] for i in range(3)])
        xx=d.collator(name,32,32)([data[i] for i in range(rank,3,2)])
        ddp.train();ref.train();ddp.zero_grad(set_to_none=True)
        (torch.nn.functional.cross_entropy(ddp(xx['inputs']),xx['labels'],reduction='sum')*2/3).backward()
        torch.nn.functional.cross_entropy(ref(x['inputs']),x['labels']).backward()
        torch.testing.assert_close(d.flat_grad(m),d.flat_grad(ref),atol=3e-5,rtol=3e-3)
        reports[name]=dict(train=tr,validation=val)
        del ref,ddp,m,opt
    # Exact epoch-boundary resume: model, momentum, scheduler and per-rank RNG.
    name='hierarchy_polarity';data=SelectData(name,size=7);history=[];best=None
    identity=dict(settings=settings,protocol='CPU test')
    states=[]
    for branch in ['whole','split']:
        d.seed_all(settings,device)
        ddp=DDP(d.make_model(name,32,32),broadcast_buffers=False)
        opt=torch.optim.SGD(ddp.parameters(),lr=.05,momentum=.9,weight_decay=.0001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
        out=Path(folder)/branch
        if rank==0:out.mkdir()
        dist.barrier();history=[];best=None
        for epoch in [1,2]:
            if branch=='split' and epoch==2:
                del ddp,opt,sched
                ddp=DDP(d.make_model(name,32,32),broadcast_buffers=False)
                opt=torch.optim.SGD(ddp.parameters(),lr=.05,momentum=.9,weight_decay=.0001)
                sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
                start,history,best=d.resume_epoch(out,ddp,opt,sched,device,identity)
                assert start==2
            lr=opt.param_groups[0]['lr']
            tr=d.run_epoch(ddp,data,name,settings,device,epoch,opt,height=32,width=32)
            sched.step();history.append(dict(epoch=epoch,learning_rate=lr,train=tr));best=dict(epoch=epoch)
            d.save_epoch(out,ddp,opt,sched,device,epoch,identity,history,best,True)
        states.append(torch.load(out/'checkpoint_last.pt',weights_only=False))
        check=d.export_and_verify(out,name,data,device,height=32,width=32)
        assert check['best']['strict_single_gpu_load']
        del ddp,opt,sched
    for key in ['model','optimizer','scheduler','rng_by_rank','sampler']:
        assert_nested_equal(states[0][key],states[1][key])
    assert states[0]['scheduler']['T_max']==50 and states[0]['scheduler']['last_epoch']==2
    if rank==0:(Path(folder)/'checks.json').write_text(json.dumps(reports))
    dist.destroy_process_group()


def test_two_rank_gloo_all_variants_and_resume(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    assert set(json.loads((tmp_path/'checks.json').read_text()))==set(d.VARIANTS)
