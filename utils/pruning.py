""" Functions that are used in prune.py """
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from enot_lite.benchmark import Benchmark
from enot_lite.type import BackendType
from fvcore.nn.flop_count import FlopCountAnalysis

from models.common import DetectMultiBackend
from utils.torch_utils import time_sync


def sample_to_n_samples(sample):
    """Function which computes the number of instances (objects to process)
    in single dataloader batch (dataloader sample)."""
    return sample[0].shape[0]


def sample_to_model_inputs(sample, device):
    """Function to map dataloader samples to model input format."""
    images = sample[0].to(device)
    images = images.float()
    images /= 255
    return (images,), {}


def count_mmac(model, dataloader, device):
    """Computes FLOPs (in MMACs)."""
    inputs, _ = sample_to_model_inputs(next(iter(dataloader)), device)
    flop_counter = FlopCountAnalysis(model=model.eval(), inputs=inputs)
    flop_counter.unsupported_ops_warnings(False)
    flop_counter.uncalled_modules_warnings(False)
    mflops = flop_counter.total() / 1e+6
    return mflops


def loss_function(model_output, sample, loss_fn, device):
    """Compute loss between model output and dataset sample."""
    labels = sample[1].to(device)
    loss, _ = loss_fn(model_output, labels)
    return loss


def measure_inference_time_torch(model, bs, size, device, warmup=50, repeat=50, number=50):
    """Compute inference time for the model"""
    inputs = torch.ones(bs, 3, size, size).to(device)

    # we want to measure time exactly as in val.py where we load model via
    # DetectMultiBackend and fuse weights
    with tempfile.TemporaryDirectory() as tmpdir:
        # save model
        ckpt_name = Path(tmpdir) / 'temp.pt'
        ckpt = {'model': deepcopy(model).half()}
        torch.save(ckpt, ckpt_name)
        # load
        model_dmb = DetectMultiBackend(ckpt_name, device=device, dnn=False, fp16=False)

    model_dmb.eval()

    times = []

    for _ in range(warmup):  # Warmup.
        with torch.no_grad():
            model_dmb(inputs)

    for i in range(repeat):
        for _ in range(number):
            with torch.no_grad():
                start = time_sync()
                model_dmb(inputs)
                end = time_sync()

            times.append(end - start)

    return np.mean(times) * 10 ** 3 / bs


def measure_inference_time_ort_cpu_single_thread(model, image_size, model_device, warmup=50, repeat=50, number=50):
    """Compute inference time using enot_lite ORT_CPU in single thread"""
    input_shape = (1, 3, image_size, image_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        # save model
        onnx_name = Path(tmpdir) / 'temp.onnx'
        torch.onnx.export(
            model=model,
            args=torch.zeros(input_shape, device=model_device),
            f=str(onnx_name),
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

        # load
        benchmark = Benchmark(
            batch_size=1,
            onnx_model=onnx_name,
            onnx_input=np.ones(input_shape, dtype=np.float32),
            backends=[BackendType.ORT_CPU],
            warmup=warmup,
            repeat=repeat,
            number=number,
            no_data_transfer=True,
            inter_op_num_threads=1,
            intra_op_num_threads=1,
        )

        benchmark.run()
        results = benchmark.results
        return results['ORT_CPU'][1]
