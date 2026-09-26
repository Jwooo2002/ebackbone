"""Train existing no-TS polarity hierarchy to a cutoff with a 100-epoch schedule."""
import argparse
import json
from pathlib import Path

from .hierarchy_polarity_training import run_model
from .v1 import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/hierarchy_polarity.n_imagenet_mini.json')
    parser.add_argument('--raw-cache',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--stop-after-epoch',type=int,default=75)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    config,locations=load_config(args.config)
    if config.epochs!=100:
        raise ValueError('Preserve the original 100-epoch cosine horizon; set cutoff with --stop-after-epoch')
    if not 1<=args.stop_after_epoch<=100:
        raise ValueError('stop-after-epoch must be between 1 and 100')
    report=run_model('hierarchy_polarity',config,**locations,raw_cache=args.raw_cache,
                     output_dir=args.output_dir,resume=args.resume,stop_after_epoch=args.stop_after_epoch)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
