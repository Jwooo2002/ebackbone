from functools import partial
import math
import numpy as np

import pytest
import torch

from ebackbone_v3 import hierarchy_polarity_training as training
from ebackbone_v3.hierarchy_polarity_models import make_model
from ebackbone_v3.hierarchy_data import collate
from tests.test_hierarchy import FakeDataset

def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


class SignedDataset(FakeDataset):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        points=item['inputs']['points']
        points[:,3]=2*(torch.arange(len(points))%2)-1
        return item


def test_partial_run_retains100_schedule_and_exact_resume(monkeypatch,tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(training,'HierarchyDataset',SignedDataset)
    monkeypatch.setattr(training,'make_model',lambda base='hierarchy',num_classes=100,height=32,width=32:
                        make_model(base,num_classes,height,width))
    monkeypatch.setattr(training,'collate',partial(collate,height=32,width=32))
    config=training.TrainConfig(epochs=100,batch_size=2,accumulation_steps=1,device='cpu',precision='float32',num_workers=0,cpu_threads=1)
    def run(folder,**kwargs):
        return training.run_model('hierarchy_polarity',config,manifest_dir='unused',dataset_root='unused',output_dir=tmp_path/folder,**kwargs)
    whole=run('whole',stop_after_epoch=3)
    first=run('resume',stop_after_epoch=1)
    resumed=run('resume',resume=True,stop_after_epoch=3)
    assert first['status']=='partial' and first['requested_segment_complete']
    assert whole['scheduler_horizon_epochs']==resumed['scheduler_horizon_epochs']==100
    assert resumed['completed_epochs']==3 and resumed['requested_stop_epoch']==3
    for record in resumed['history']:
        expected=.05*(1+math.cos(math.pi*(record['epoch']-1)/100))/2
        assert record['learning_rate']==pytest.approx(expected)
    states=[torch.load(tmp_path/f/'checkpoint_last.pt',weights_only=False) for f in ['whole','resume']]
    for key in ['model','optimizer','scheduler','rng']:
        assert_nested_equal(states[0][key],states[1][key])
    assert states[1]['scheduler']['T_max']==100 and states[1]['scheduler']['last_epoch']==3
    assert states[1]['optimizer']['state']  # Momentum buffers preserved.
    for kind in ['best','last']:
        c=torch.load(tmp_path/'resume'/f'checkpoint_{kind}.pt',weights_only=False)
        assert {'model','optimizer','scheduler','rng','identity','history','best','epoch'}<=set(c)
    assert resumed['checkpoint_verification']['logits_bit_exact']
    with pytest.raises(ValueError,match='overwrite'):run('whole',stop_after_epoch=3)
    with pytest.raises(ValueError,match='unknown'):training.run_model('hierarchy_ts',config,manifest_dir='unused',dataset_root='unused',output_dir=tmp_path/'bad')


def test_epoch40_archives_preserve_entire_checkpoint(tmp_path):
    for kind,epoch in [('best',39),('last',40)]:
        torch.save({'epoch':epoch,'model':{'weight':torch.arange(4)},'optimizer':{'momentum':.9},
                    'scheduler':{'T_max':100,'last_epoch':epoch},'rng':{'torch':torch.get_rng_state()},
                    'identity':{'architecture':'hierarchy_polarity'},'history':[{'epoch':epoch}],'best':{'epoch':39}},
                   tmp_path/f'checkpoint_{kind}.pt')
    training.capture_epoch40(tmp_path)
    for kind in ['best','last']:
        assert (tmp_path/f'checkpoint_{kind}.pt').read_bytes()==(tmp_path/f'checkpoint_{kind}_through40.pt').read_bytes()


def test_cli_rejects_compressed_cosine_schedule(monkeypatch,tmp_path):
    from ebackbone_v3 import hierarchy_polarity_train as cli
    monkeypatch.setattr(cli,'load_config',lambda _: (training.TrainConfig(epochs=75),{}))
    with pytest.raises(ValueError,match='100-epoch'):
        cli.main(['--output-dir',str(tmp_path/'never-started')])
    assert not (tmp_path/'never-started').exists()
