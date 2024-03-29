# standard lib
import random
import warnings

# 3rd-party lib
import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# mm lib
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, build_optimizer, build_runner
from mmcls.core import DistOptimizerHook
# from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger

from mmcls.core import EvalHook, DistEvalHook
from ..core import CrossArchEvalHook, DistCrossArchEvalHook

# TODO import optimizer hook from mmcv and delete them from mmcls
try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn('DeprecationWarning: FP16OptimizerHook from mmcls will be '
                  'deprecated. Please install mmcv>=1.1.4.')
    from mmcls.core import Fp16OptimizerHook

# gaia lib
from gaiavision.core import ManipulateArchHook
# registry new file
import gaiacls
from gaiacls.datasets import build_dataloader, build_dataset

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model,
                train_sampler,
                val_sampler,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                device='cuda',
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            repeated_aug=cfg.repeated_aug,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cuda':
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        elif device == 'cpu':
            model = MMDataParallel(model.cpu())
        else:
            raise ValueError(F'unsupported device name {device}.')
    #build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # add hook for architecture manipulation
    # manipulate_arch_hook = ManipulateArchHook(train_sampler)
    # runner.register_hook(manipulate_arch_hook)
    if cfg.manipulate_arch:
        manipulate_arch_hook = ManipulateArchHook(train_sampler)
        runner.register_hook(manipulate_arch_hook)

    cfg.gpu_collect=cfg.get('gpu_collect', False)
    # register eval hooks
    if validate:
        if cfg.manipulate_arch:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=64,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistCrossArchEvalHook if distributed else CrossArchEvalHook
            runner.register_hook(eval_hook(val_dataloader, val_sampler, **eval_cfg))
        else:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=64,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, gpu_collect=cfg.gpu_collect, **eval_cfg))


    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)
