# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Prune a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python prune.py --data data/coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from enot.pruning import calibrate_and_prune_model_equal
from enot.pruning import calibrate_and_prune_model_global_wrt_metric_drop
from enot.pruning import calibrate_and_prune_model_optimal

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, increment_path,
                           labels_to_class_weights, methods, print_args)
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.pruning import count_mmac
from utils.pruning import loss_function
from utils.pruning import measure_inference_time_ort_cpu_single_thread
from utils.pruning import measure_inference_time_torch
from utils.pruning import sample_to_model_inputs
from utils.pruning import sample_to_n_samples

from utils.torch_utils import de_parallel, select_device, fix_model_compatibility_between_version

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))


def prune(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.cfg, \
        opt.workers
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    if loggers.wandb:
        data_dict = loggers.wandb.data_dict

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check

    # Model
    check_suffix(weights, '.pt')  # check weights
    if not weights.endswith('.pt'):
        raise ValueError(f'wrong checkpoint name {weights}')

    weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

    if ckpt.get('ema'):
        model = ckpt['ema']
    else:
        model = ckpt['model']

    model.float().to(device)
    model = fix_model_compatibility_between_version(model)
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              workers=workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size * 2,
                                   gs,
                                   single_cls,
                                   rect=True,
                                   hyp=hyp,
                                   workers=workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    callbacks.run('on_pretrain_routine_end')

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Prepare for prune
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                'Starting pruning...')

    # Save original model
    if opt.save_before_prune:
        ckpt = {
            'epoch': -1,
            'best_fitness': 0.0,
            'model': deepcopy(de_parallel(model)).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
            'date': datetime.now().isoformat()}

        # Save last, best and delete
        torch.save(ckpt, w / 'original_model.pt')

    # Test model before pruning
    results, maps, _ = val.run(data_dict,
                               batch_size=batch_size * 2,
                               imgsz=imgsz,
                               model=model.eval(),
                               single_cls=single_cls,
                               dataloader=val_loader,
                               save_dir=save_dir,
                               plots=False,
                               callbacks=callbacks,
                               compute_loss=compute_loss)

    model.train()
    loss_fn = partial(loss_function, loss_fn=compute_loss, device=device)
    pruning_hyps = hyp['pruning']
    pruning_mode = pruning_hyps['pruning_mode']
    calibration_steps = pruning_hyps['calibration_steps']
    calibration_epochs = pruning_hyps['calibration_epochs']
    sample_to_inputs = partial(sample_to_model_inputs, device=device)

    cost_function = partial(
        count_mmac,
        dataloader=train_loader,
        device=device
    )
    latency_units = 'MMACs'
    original_model_cost = cost_function(model.eval())

    LOGGER.info(f"Original model cost: {original_model_cost:.2f} {latency_units}")
    model.train()

    if pruning_mode == 'equal':
        equal_pruning_mode_hyps = pruning_hyps[pruning_mode]
        pruned_model = calibrate_and_prune_model_equal(
            model=model,
            dataloader=train_loader,
            loss_function=loss_fn,
            pruning_ratio=equal_pruning_mode_hyps['pruning_ratio'],
            finetune_bn=True,
            sample_to_n_samples=sample_to_n_samples,
            sample_to_model_inputs=sample_to_inputs,
            calibration_epochs=calibration_epochs,
            calibration_steps=calibration_steps,
            show_tqdm=True,
        )
    elif pruning_mode == 'optimal':
        optimal_pruning_mode_hyps = pruning_hyps[pruning_mode]
        if opt.n_search_steps is not None:
            optimal_pruning_mode_hyps['n_search_steps'] = opt.n_search_steps

        latency_type = optimal_pruning_mode_hyps['latency_type']

        if latency_type == 'time':
            time_hyps = optimal_pruning_mode_hyps['time']
            if time_hyps['backend'] == 'ort_cpu':
                cost_function = partial(
                    measure_inference_time_ort_cpu_single_thread,
                    image_size=imgsz,
                    model_device=device,
                    warmup=time_hyps['warmup'],
                    repeat=time_hyps['repeat'],
                    number=time_hyps['number'],
                )
            elif time_hyps['backend'] == 'torch':
                cost_function = partial(
                    measure_inference_time_torch,
                    bs=opt.batch_size,
                    size=imgsz,
                    device=device,
                    warmup=time_hyps['warmup'],
                    repeat=time_hyps['repeat'],
                    number=time_hyps['number'],
                )

            latency_units = 'ms'

        elif latency_type != 'flops':
            raise ValueError(f'Unknown latency type {latency_type}, should be one of'
                             f'["flops", "time"]')

        original_model_cost = cost_function(model.eval())
        LOGGER.info(f"Original model cost: {original_model_cost:.2f} {latency_units}")

        if opt.target_latency_fraction is not None:
            optimal_pruning_mode_hyps['target_latency_fraction'] = opt.target_latency_fraction

        target_latency = original_model_cost * optimal_pruning_mode_hyps['target_latency_fraction']
        LOGGER.info(f"Target model cost: {target_latency:.2f} {latency_units}")

        if optimal_pruning_mode_hyps['target_latency']:
            target_latency = optimal_pruning_mode_hyps['target_latency']

        model.train()

        pruned_model = calibrate_and_prune_model_optimal(
            model=model,
            dataloader=train_loader,
            loss_function=loss_fn,
            latency_calculation_function=cost_function,
            target_latency=target_latency,
            finetune_bn=True,
            calibration_steps=calibration_steps,
            calibration_epochs=calibration_epochs,
            n_search_steps=optimal_pruning_mode_hyps['n_search_steps'],
            sample_to_model_inputs=sample_to_inputs,
            sample_to_n_samples=sample_to_n_samples,
            show_tqdm=True,
            latency_penalty=optimal_pruning_mode_hyps['latency_penalty'],
        )

    elif pruning_mode == 'global_wrt_metric':
        global_pruning_mode_hyps = pruning_hyps[pruning_mode]

        def eval_map(pruned_model):
            results, maps, _ = val.run(
                data_dict,
                batch_size=batch_size * 2,
                imgsz=imgsz,
                model=pruned_model,
                single_cls=single_cls,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=False,
                compute_loss=compute_loss,
            )
            pruned_model.train()
            # results is sequence of mean_precision, mean_recall, map50, map and some detection losses
            # here we optimize map firstly
            return results[3]

        pruned_model = calibrate_and_prune_model_global_wrt_metric_drop(
            model=model.train(),
            dataloader=train_loader,
            loss_function=loss_fn,
            validation_function=eval_map,
            maximal_acceptable_metric_drop=global_pruning_mode_hyps['maximal_acceptable_metric_drop'],
            minimal_channels_to_prune=global_pruning_mode_hyps['minimal_channels_to_prune'],
            maximal_channels_to_prune=global_pruning_mode_hyps['maximal_channels_to_prune'],
            channel_step_to_search=global_pruning_mode_hyps['channel_step_to_search'],
            finetune_bn=True,
            calibration_steps=calibration_steps,
            calibration_epochs=calibration_epochs,
            sample_to_model_inputs=sample_to_inputs,
            sample_to_n_samples=sample_to_n_samples,
            show_tqdm=True,
        )
    else:
        raise ValueError(f'No such pruning mode:{pruning_mode}. Possible values: equal, global_wrt_metric, optimal.')

    pruned_model.to(device)
    pruned_model_cost = cost_function(pruned_model.eval())

    LOGGER.info(f"Pruned model cost: {pruned_model_cost:.2f} {latency_units}")
    LOGGER.info(f"Acceleration x{original_model_cost / pruned_model_cost:.4f} after pruning")

    LOGGER.info(f"Eval pruned model")
    # eval pruned model
    results, maps, _ = val.run(data_dict,
                               batch_size=batch_size * 2,
                               imgsz=imgsz,
                               model=pruned_model,
                               single_cls=single_cls,
                               dataloader=val_loader,
                               save_dir=save_dir,
                               plots=False,
                               compute_loss=compute_loss)

    if opt.save_after_prune:
        ckpt = {
            'epoch': -1,
            'best_fitness': 0.0,
            'model': deepcopy(de_parallel(pruned_model)).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
            'date': datetime.now().isoformat(),
            'map095': results[3],
        }

        # Save last, best and delete
        torch.save(ckpt, w / 'pruned_model.pt')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/prune', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument(
        '--save_before_prune',
        action='store_true',
        help='Save checkpoint for original model.',
    )
    parser.add_argument(
        '--save_after_prune',
        type=bool,
        default=True,
        help='Save checkpoint for pruned model.',
    )
    parser.add_argument(
        '--n-search-steps',
        type=int,
        default=None,
        help='Number of steps for optimal architecture search.'
             'Default value is None, which means that value from hyp will be used.'
    )
    parser.add_argument(
        '--target-latency-fraction',
        type=float,
        default=None,
        help='Fraction of target latency for optimal architecture.'
             'Default value is None, which means that value from hyp will be used.'
    )

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    print_args(vars(opt))
    check_git_status()
    check_requirements(exclude=['thop'])

    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    prune(opt.hyp, opt, device, callbacks)


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
