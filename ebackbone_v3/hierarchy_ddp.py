"""Isolated two-rank batch64/50-epoch study; existing trainers are untouched."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from .hierarchy_models import HierarchyV1
from .hierarchy_ts_models import HierarchyTSV1
from .hierarchy_ts_residual_models import HierarchyTSResidualV2
from .hierarchy_ts_confidence_models import HierarchyTSConfidenceV3
from .hierarchy_ablation_models import make_model as ablation_model
from .hierarchy_polarity_models import make_model as polarity_model
from .hierarchy_data import HierarchyDataset, collate as point_collate
from .hierarchy_ts_data import HierarchyTSDataset, collate as ts_collate
from .v1_training import atomic_json, atomic_checkpoint, sha256_file, seed_worker

VARIANTS = ('hierarchy', 'hierarchy_ts', 'hierarchy_ts_residual', 'hierarchy_ts_confidence',
            'hierarchy_local', 'hierarchy_self_control', 'hierarchy_early_skip', 'hierarchy_polarity')
TS_VARIANTS = VARIANTS[1:4]


def make_model(name, height=480, width=640):
    factories = dict(hierarchy=HierarchyV1, hierarchy_ts=HierarchyTSV1,
                     hierarchy_ts_residual=HierarchyTSResidualV2,
                     hierarchy_ts_confidence=HierarchyTSConfidenceV3)
    if name in factories:
        return factories[name](height=height, width=width)
    if name == 'hierarchy_polarity':
        return polarity_model('hierarchy', height=height, width=width)
    if name in VARIANTS:
        return ablation_model(name, height=height, width=width)
    raise ValueError(name)


def collator(name, height=480, width=640):
    return partial(ts_collate if name in TS_VARIANTS else point_collate, height=height, width=width)


class ExactBatchShard(Sampler):
    """Partition global batches without repeating or dropping sample indices.

    Train batches are strided across ranks. The final nonempty global batch
    can have unequal local sizes; loss scaling uses its actual global count.
    Validation can have unequal numbers of batches and uses the unwrapped model.
    """
    def __init__(self, size, rank, world_size, batch_size, seed, epoch, train):
        if size < world_size or not 0 <= rank < world_size:
            raise ValueError('each training rank needs at least one sample')
        order = (torch.randperm(size, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
                 if train else list(range(size)))
        self.batches = []
        if train:
            chunks = [order[i:i+world_size*batch_size] for i in range(0,size,world_size*batch_size)]
            # Avoid a rank with no backward on an extremely small tail, without
            # exceeding local batch_size: rebalance the preceding global batch.
            if len(chunks)>1 and len(chunks[-1])<world_size:
                need=world_size-len(chunks[-1])
                chunks[-1]=chunks[-2][-need:]+chunks[-1]
                chunks[-2]=chunks[-2][:-need]
            self.global_sizes=[len(c) for c in chunks]
            self.batches=[c[rank::world_size] for c in chunks]
        else:
            local=order[rank::world_size]
            self.batches=[local[i:i+batch_size] for i in range(0,len(local),batch_size)]
            self.global_sizes=[]
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)


def gather(value):
    out=[None]*dist.get_world_size()
    dist.all_gather_object(out,value)
    return out


def seed_all(settings, device):
    seed=settings['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(settings['cpu_threads'])
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        assert torch.cuda.is_bf16_supported()
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def rng_state(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def flat_grad(model):
    assert all(p.grad is not None for p in model.parameters()), 'disconnected trainable parameter'
    result=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    assert torch.isfinite(result).all() and result.abs().sum()>0
    return result


def check_gradient_average(ddp,inputs,labels,global_size,device):
    """Compare DDP backward with explicitly all-reduced local gradients."""
    ddp.zero_grad(set_to_none=True)
    with ddp.no_sync(),amp(device):
        loss=torch.nn.functional.cross_entropy(ddp(inputs).float(),labels,reduction='sum')
        loss=loss*dist.get_world_size()/global_size
        loss.backward()
    expected=flat_grad(ddp.module).clone()
    dist.all_reduce(expected);expected/=dist.get_world_size()
    ddp.zero_grad(set_to_none=True)
    with amp(device):
        loss=torch.nn.functional.cross_entropy(ddp(inputs).float(),labels,reduction='sum')
        loss=loss*dist.get_world_size()/global_size
    loss.backward()
    actual=flat_grad(ddp.module)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
    error=float((actual-expected).abs().max())
    ddp.zero_grad(set_to_none=True)
    return error


def run_epoch(ddp,data,name,settings,device,epoch,optimizer=None,probe=False,height=480,width=640):
    train=optimizer is not None
    rank=dist.get_rank();world=dist.get_world_size()
    model=ddp.module
    model.train(train)
    sampler=ExactBatchShard(len(data),rank,world,settings['batch_size'],settings['seed'],epoch,train)
    batches=DataLoader(data,batch_sampler=sampler,collate_fn=collator(name,height,width),
                       num_workers=settings['num_workers'],worker_init_fn=seed_worker,
                       generator=torch.Generator().manual_seed(settings['seed']+epoch),
                       pin_memory=device.type=='cuda')
    ids=[];values=torch.zeros(4,dtype=torch.float64,device=device);grad_checks=[]
    dist.barrier();synchronize(device);started=time.perf_counter()
    for index,batch in enumerate(batches):
        assert set(batch['splits'])=={data.split} and set(batch['source_splits'])=={'train'}
        inputs={k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()}
        labels=batch['labels'].to(device,non_blocking=True)
        if hasattr(model,'capture_gate_stats'):model.capture_gate_stats=index==0
        if train:
            optimizer.zero_grad(set_to_none=True)
            if probe and index==0:
                error=check_gradient_average(ddp,inputs,labels,sampler.global_sizes[index],device)
        with torch.set_grad_enabled(train),amp(device):
            # Evaluation bypasses DDP collectives: no padded validation examples,
            # no deadlock when rank batch counts differ. GN has no running stats.
            logits=(ddp if train else model)(inputs)
            loss=torch.nn.functional.cross_entropy(logits.float(),labels,reduction='sum')
        if device.type=='cuda':assert logits.dtype==torch.bfloat16
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite CE')
        if train:
            (loss*world/sampler.global_sizes[index]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            if probe:
                gradient=flat_grad(model)
                digest=hashlib.sha256(gradient.cpu().numpy().tobytes()).hexdigest()
                assert len(set(gather(digest)))==1,'DDP gradients differ between ranks'
                stage={n:sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
                       for n,m in model.named_children() if list(m.parameters())}
                if index>=2:assert all(v>0 for v in stage.values()),stage
                grad_checks.append(dict(step=index+1,stage_l1=stage,gradient_sha256=digest,
                                        average_error=error if index==0 else None))
            optimizer.step()
            if rank==0 and (index==0 or (index+1)%100==0):
                print(json.dumps(dict(event='optimizer_step',variant=name,epoch=epoch,step=index+1,
                                      global_batch=sampler.global_sizes[index],loss_rank0=float(loss)/len(labels))),flush=True)
        predictions=logits.detach().topk(5,dim=1).indices
        values+=torch.stack([loss.detach().double(),(predictions[:,0]==labels).sum(),
                             (predictions==labels[:,None]).any(1).sum(),torch.tensor(len(labels),device=device)])
        ids.extend(batch['sample_ids'])
    synchronize(device)
    seconds=torch.tensor(time.perf_counter()-started,device=device,dtype=torch.float64)
    dist.all_reduce(seconds,op=dist.ReduceOp.MAX);dist.all_reduce(values)
    shards=gather(ids);flat=[s for shard in shards for s in shard]
    expected=[r.sample_id for r in data.rows]
    assert len(flat)==len(data)==int(values[3]) and len(set(flat))==len(flat)
    assert set(flat)==set(expected), 'shards must exactly cover selected split'
    canonical=hashlib.sha256(''.join(s+'\n' for s in expected).encode()).hexdigest()
    result=dict(loss=float(values[0]/values[3]),top1=float(values[1]/values[3]),
                top5=float(values[2]/values[3]),samples=len(flat),seconds=float(seconds),
                samples_per_second=len(flat)/float(seconds),sample_order_sha256=canonical,
                rank_sample_counts=[len(s) for s in shards],duplicates=0,missing=0,
                rank_order_sha256=[hashlib.sha256(''.join(s+'\n' for s in shard).encode()).hexdigest() for shard in shards],
                optimizer_steps=len(sampler) if train else 0)
    if probe:result['gradient_checks']=grad_checks
    return result


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_epoch(output,ddp,optimizer,scheduler,device,epoch,identity,history,best,improved):
    states=gather(rng_state(device))
    if dist.get_rank()==0:
        checkpoint=dict(identity=identity,epoch=epoch,model=cpu_state(ddp.module),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),rng_by_rank=states,history=history,best=best,
                        sampler=dict(seed=identity['settings']['seed'],next_epoch=epoch+1,world_size=dist.get_world_size()))
        if improved:atomic_checkpoint(output/'checkpoint_best.pt',checkpoint)
        atomic_checkpoint(output/'checkpoint_last.pt',checkpoint)
        atomic_json(output/'history.json',history)
    dist.barrier()


def resume_epoch(output,ddp,optimizer,scheduler,device,identity):
    c=torch.load(output/'checkpoint_last.pt',map_location='cpu',weights_only=False)
    assert c['identity']==identity and len(c['rng_by_rank'])==dist.get_world_size()
    ddp.module.load_state_dict(c['model'],strict=True)
    optimizer.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
    restore_rng(c['rng_by_rank'][dist.get_rank()],device)
    return c['epoch']+1,c['history'],c['best']


def export_and_verify(output,name,data,device,height=480,width=640):
    """Rank0 reloads each checkpoint in a plain single-device model."""
    checks={}
    if dist.get_rank()==0:
        inputs={k:v.to(device) for k,v in collator(name,height,width)([data[0]])['inputs'].items()}
        for kind in ('best','last'):
            path=output/f'checkpoint_{kind}.pt'
            c=torch.load(path,map_location='cpu',weights_only=False)
            assert not any(k.startswith('module.') for k in c['model'])
            a=make_model(name,height,width).to(device).eval();a.load_state_dict(c['model'],strict=True)
            b=make_model(name,height,width).to(device).eval();b.load_state_dict(c['model'],strict=True)
            with torch.no_grad(),amp(device):assert torch.equal(a(inputs),b(inputs))
            backbone={k:v for k,v in c['model'].items() if not k.startswith('classifier.')}
            conditioning=({k:v for k,v in c['model'].items() if k.startswith('classifier.')}
                          if name=='hierarchy_ts_confidence' else {})
            export=dict(backbone=backbone,conditioning_classifier=conditioning,variant=name,epoch=c['epoch'],
                        checkpoint_sha256=sha256_file(path),identity=c['identity'],
                        note='V3 shared classifier is needed to compute its confidence gate; supplied separately.' if conditioning else '')
            exported=output/f'backbone_{kind}.pt';atomic_checkpoint(exported,export)
            saved=torch.load(exported,map_location='cpu',weights_only=False)
            assert saved['backbone'].keys()==backbone.keys()
            assert all(torch.equal(saved['backbone'][k],v) for k,v in backbone.items())
            checks[kind]=dict(strict_single_gpu_load=True,no_ddp_prefix=True,repeat_reload_logits_bit_exact=True,
                              checkpoint_sha256=sha256_file(path),backbone_sha256=sha256_file(exported))
            del a,b,c
    return gather(checks)[0]


def memory(device):
    return dict(rank=dist.get_rank(),allocated_bytes=torch.cuda.max_memory_allocated(device),
                reserved_bytes=torch.cuda.max_memory_reserved(device),device_name=torch.cuda.get_device_name(device)) if device.type=='cuda' else dict(rank=dist.get_rank(),allocated_bytes=0,reserved_bytes=0)


def execute(args):
    local=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    assert dist.get_world_size()==2
    plan=json.loads((args.study/'plan.json').read_text());settings=plan['settings']
    expected_uuid=plan['cuda_visible_devices'].split(',')[local].removeprefix('GPU-')
    assert str(torch.cuda.get_device_properties(device).uuid)==expected_uuid
    assert settings['epochs']==50 and settings['batch_size']==32 and settings['accumulation_steps']==1
    assert settings['learning_rate']==.05 and plan['global_batch_size']==64
    seed_all(settings,device)
    entry=next(v for v in plan['variants'] if v['name']==args.variant)
    source={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    assert source==plan['source_sha256'], 'frozen source changed'
    dataset_class=HierarchyTSDataset if args.variant in TS_VARIANTS else HierarchyDataset
    locations=plan['data']
    train=dataset_class(**locations,split='train',limit=192 if args.mode=='preflight' else None,seed=settings['seed'])
    validation=dataset_class(**locations,split='validation',limit=65 if args.mode=='preflight' else None,seed=settings['seed'])
    assert train.manifest_sha256==entry['train_manifest_sha256']
    assert validation.manifest_sha256==entry['validation_manifest_sha256']
    assert not {r.sample_id for r in train.rows}.intersection(r.sample_id for r in validation.rows)
    output=args.study/('preflight' if args.mode=='preflight' else 'runs')/args.variant
    if dist.get_rank()==0:
        if args.resume:
            assert (output/'checkpoint_last.pt').is_file()
        else:
            assert not output.exists(),'refusing existing output'
            output.mkdir(parents=True)
    dist.barrier()
    model=make_model(args.variant).to(device)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=settings['learning_rate'],momentum=settings['momentum'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    identity=dict(protocol=plan['protocol'],variant=args.variant,settings=settings,world_size=2,global_batch_size=64,
                  train_manifest_sha256=train.manifest_sha256,validation_manifest_sha256=validation.manifest_sha256,
                  source_sha256=source,raw_cache=locations['raw_cache'],mode=args.mode,torch_version=torch.__version__,numpy_version=np.__version__)
    first=1;history=[];best=None
    if args.resume:first,history,best=resume_epoch(output,ddp,optimizer,scheduler,device,identity)
    if dist.get_rank()==0:atomic_json(output/'config.json',identity)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first,2 if args.mode=='preflight' else 51):
        lr=optimizer.param_groups[0]['lr']
        training=run_epoch(ddp,train,args.variant,settings,device,epoch,optimizer,probe=args.mode=='preflight')
        evaluation=run_epoch(ddp,validation,args.variant,settings,device,epoch)
        scheduler.step()
        improved=best is None or (evaluation['top1'],-evaluation['loss'])>(best['top1'],-best['loss'])
        if improved:best=dict(epoch=epoch,**evaluation)
        history.append(dict(epoch=epoch,learning_rate=lr,train=training,validation=evaluation,gpu_memory_by_rank=gather(memory(device))))
        save_epoch(output,ddp,optimizer,scheduler,device,epoch,identity,history,best,improved)
        if dist.get_rank()==0:print(json.dumps(dict(event='epoch_complete',variant=args.variant,**history[-1])),flush=True)
    # Preflight also checks stored state against the actual trained model, not
    # merely whether two reloads agree with each other.
    c=torch.load(output/'checkpoint_last.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.detach().cpu(),c['model'][k]) for k,v in model.state_dict().items())
    reference={k:v.to(device) for k,v in collator(args.variant)([train[0]])['inputs'].items()}
    model.eval()
    with torch.no_grad(),amp(device):expected=model(reference)
    restored=make_model(args.variant).to(device).eval();restored.load_state_dict(c['model'],strict=True)
    with torch.no_grad(),amp(device):assert torch.equal(expected,restored(reference))
    del restored,c,reference
    checks=export_and_verify(output,args.variant,train,device)
    peak=gather(memory(device))
    if args.mode=='preflight':
        assert all(p['reserved_bytes'] < torch.cuda.get_device_properties(device).total_memory*.9 for p in peak)
    if dist.get_rank()==0:
        atomic_json(output/'report.json',dict(status='PASS' if args.mode=='preflight' else 'complete',identity=identity,
                    completed_epochs=history[-1]['epoch'],best=best,final=history[-1]['validation'],
                    checkpoint_verification=checks,trained_model_reload_bit_exact=True,
                    train_samples_per_second=sum(r['train']['samples'] for r in history)/sum(r['train']['seconds'] for r in history),
                    gpu_memory_by_rank=peak,final_test_accessed=False,backbone_exported=True))
    dist.barrier();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--mode',choices=['preflight','train'],required=True)
    parser.add_argument('--resume',action='store_true')
    execute(parser.parse_args())

if __name__=='__main__':main()
