import pytest
import torch

from ebackbone_v3.hierarchy_models import PointToVoxel, profile_macs as baseline_profile
from ebackbone_v3.hierarchy_data import collate
from ebackbone_v3.hierarchy_ts_models import profile_macs as ts_profile
from ebackbone_v3.hierarchy_polarity_models import BASE_MODELS, PolarityPointToVoxel, make_model, profile_macs
from tests.test_hierarchy import sample


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def signed_sample(index):
    s=sample(index)
    s['inputs']['points'][:,3]=2*(torch.arange(len(s['inputs']['points']))%2)-1
    return s


def test_collision_separation_mass_conservation_and_gradient_isolation():
    points=torch.tensor([[0.,0.,.1,-1.],[0.,0.,.1,1.]])
    inputs={'points':points,'voxel_lower':torch.tensor([0,0]),'voxel_upper':torch.tensor([64,64]),
            'alpha':torch.tensor([.25,.25]),'event_counts':torch.tensor([2])}
    features=torch.tensor([2.,6.])[:,None].expand(-1,32).clone().requires_grad_()
    sep=PolarityPointToVoxel(32,32)(features,inputs)
    mixed=PointToVoxel(32,32)(features,inputs)
    assert sep.shape==(1,66,8,8,8) and sep.dtype==torch.float32
    assert mixed[0,0,0,0,0]==4
    assert sep[0,0,0,0,0]==2 and sep[0,33,0,0,0]==6
    neg_mass,pos_mass=sep[:,32].expm1(),sep[:,65].expm1()
    torch.testing.assert_close(neg_mass.sum(),torch.tensor(1.))
    torch.testing.assert_close(pos_mass.sum(),torch.tensor(1.))
    torch.testing.assert_close(neg_mass+pos_mass,mixed[:,32].expm1())
    assert neg_mass[0,0,0,0].item()==pytest.approx(.75)
    assert neg_mass[0,1,0,0].item()==pytest.approx(.25)
    assert torch.count_nonzero(sep[:,:,:,1:,1:])==0
    sep[:,:32].sum().backward()
    assert torch.all(features.grad[0]>0) and torch.all(features.grad[1]==0)
    flipped={**inputs,'points':points*torch.tensor([1.,1.,1.,-1.])}
    alternate=PolarityPointToVoxel(32,32)(features.detach(),flipped)
    assert torch.equal(alternate[:,:33],sep[:,33:]) and torch.equal(alternate[:,33:],sep[:,:33])
    assert torch.equal(PointToVoxel(32,32)(features.detach(),flipped),mixed)


@pytest.mark.parametrize('sign',[-1.,1.])
def test_missing_polarity_and_terminal_bin(sign):
    inputs={'points':torch.tensor([[1.,1.,1.,sign]]),'voxel_lower':torch.tensor([511]),
            'voxel_upper':torch.tensor([511]),'alpha':torch.tensor([0.]),'event_counts':torch.tensor([1])}
    grid=PolarityPointToVoxel(32,32)(torch.ones(1,32),inputs)
    occupied=grid[:,:33] if sign<0 else grid[:,33:]
    empty=grid[:,33:] if sign<0 else grid[:,:33]
    assert torch.count_nonzero(empty)==0 and torch.isfinite(grid).all()
    assert occupied[0,0,7,7,7]==1
    torch.testing.assert_close(occupied[:,32].expm1().sum(),torch.tensor(1.))
    inputs['points'][:,3]=0
    with pytest.raises(ValueError,match='signed'):PolarityPointToVoxel(32,32)(torch.ones(1,32),inputs)


def test_packed_sample_isolation():
    samples=[signed_sample(0),signed_sample(3)]
    generator=torch.Generator().manual_seed(7)
    values=[torch.rand(len(s['inputs']['points']),32,generator=generator) for s in samples]
    module=PolarityPointToVoxel(32,32)
    individual=torch.cat([module(f,s['inputs']) for f,s in zip(values,samples)])
    packed=module(torch.cat(values),collate(samples,height=32,width=32)['inputs'])
    assert torch.equal(packed,individual)


@pytest.mark.parametrize('base',tuple(BASE_MODELS))
def test_only_aggregation_and_fusion_change_with_bf16_gradients_and_reload(base,tmp_path):
    torch.manual_seed(42);original=BASE_MODELS[base](height=32,width=32)
    rng=torch.get_rng_state()
    torch.manual_seed(42);model=make_model(base,height=32,width=32)
    assert torch.equal(torch.get_rng_state(),rng)
    assert type(model) is type(original)
    assert model.forward_features.__func__ is original.forward_features.__func__
    changed=[]
    for key,value in original.state_dict().items():
        if not torch.equal(value,model.state_dict()[key]):changed.append(key)
    assert changed==['voxel_projection.0.weight']
    assert model.voxel_projection[0].weight.shape==(32,66,1,1,1)
    inputs=collate([signed_sample(0),signed_sample(1)],height=32,width=32)['inputs']
    if base!='hierarchy':inputs['time_surface']=torch.rand(2,2,32,32)
    model.train();optimizer=torch.optim.SGD(model.parameters(),lr=.01,weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            logits=model(inputs)
            loss=torch.nn.functional.cross_entropy(logits.float(),torch.tensor([0,1]))
        assert logits.shape==(2,100) and logits.dtype==torch.bfloat16 and torch.isfinite(loss)
        loss.backward()
        for name,module in model.named_children():
            params=list(module.parameters())
            if not params:continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),name
            grad=sum(float(p.grad.abs().sum()) for p in params)
            expected_zero=step==0 and (name=='ts_encoder' or name=='confidence_control')
            if expected_zero:assert grad==0,name
            else:assert grad>0,name
        weight_grad=model.voxel_projection[0].weight.grad
        assert weight_grad[:,:33].abs().sum()>0 and weight_grad[:,33:].abs().sum()>0
        optimizer.step()
    path=tmp_path/'checkpoint.pt';torch.save(model.state_dict(),path)
    restored=make_model(base,height=32,width=32).eval()
    restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    model.eval()
    with torch.no_grad():assert torch.equal(model(inputs),restored(inputs))
    with pytest.raises(RuntimeError,match='size mismatch'):model.load_state_dict(original.state_dict(),strict=True)


@pytest.mark.parametrize('base',tuple(BASE_MODELS))
def test_native_shapes_and_exact_added_cost(base):
    model=make_model(base)
    result=profile_macs(model)
    reference=(baseline_profile if base=='hierarchy' else ts_profile)(BASE_MODELS[base]())
    assert result['parameters']-reference['parameters']==1056
    assert result['macs']-reference['macs']==162201600
    shapes=result['shapes_batch1']
    assert shapes['point']==[116790,32]
    assert shapes['point_to_voxel']==[1,66,8,120,160]
    assert shapes['voxel_projection']==[1,32,8,120,160]
    assert shapes['voxel_stage1']==[1,64,8,60,80]
    assert shapes['voxel_stage2']==[1,128,8,30,40]
    assert shapes['temporal_collapse']==[1,128,30,40]
    assert shapes['frame_stage']==[1,256,15,20]
