import argparse
from functools import partial

import torch
from enot.quantization import DefaultQuantizationDistiller
from enot.quantization import OpenvinoFakeQuantizedModel
from enot.quantization import TrtFakeQuantizedModel
from onnx2torch import convert

from utils.dataloaders import create_dataloader
from utils.general import check_dataset


def sample_to_model_inputs(x, device):
    # x[0] is the first item from dataloader sample. Sample is a tuple where 0'th element is a tensor with images.
    x = x[0]

    # Model is on CUDA, so input images should also be on CUDA.
    x = x.to(device)

    # Converting tensor from int8 to float data type.
    x = x.float()

    # YOLOv5 image normalization (0-255 to 0-1 normalization)
    x /= 255
    return (x,), {}


def main(opt):
    if isinstance(opt.device, str):
        device = torch.device(opt.device)
    IMG_SHAPE = (opt.batch_size, 3, opt.imgsz, opt.imgsz)

    data = check_dataset(opt.data)

    valid_dataloader = create_dataloader(
        path=data['val'],
        imgsz=opt.imgsz,
        batch_size=opt.batch_size,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=False,
        workers=opt.workers,
        hyp=opt.hyp,
    )[0]

    regular_model = convert(opt.weights).to(device)
    regular_model.eval()

    # Please consider to specify `quantization_scheme` for `OpenvinoFakeQuantizedModel`,
    # quantization scheme can affect the perfomance of the quantized model.
    # See for details: https://enot-autodl.rtd.enot.ai/en/stable/reference_documentation/quantization.html#enot.quantization.TrtFakeQuantizedModel

    if opt.backend == 'openvino':
        fake_quantized_model = OpenvinoFakeQuantizedModel(regular_model).to(device)
    elif opt.backend == 'tensorrt':
        fake_quantized_model = TrtFakeQuantizedModel(regular_model).to(device)
    else:
        ValueError('Invalid backend argument!')

    # TODO: maybe use train dataloader
    dist = DefaultQuantizationDistiller(
        quantized_model=fake_quantized_model,
        dataloader=valid_dataloader,
        sample_to_model_inputs=partial(sample_to_model_inputs, device=device),
        device=device,
        logdir=opt.log_dir,
        verbose=2,
    )

    dist.distill()

    fake_quantized_model.to('cpu')
    fake_quantized_model.enable_quantization_mode(True)
    fake_quantized_model.to('cpu')

    torch.onnx.export(
        model=fake_quantized_model,
        args=torch.ones(*IMG_SHAPE),
        f=opt.weights.replace('.onnx', '_quant.onnx'),
        input_names=['images'],
        output_names=['output'],
        opset_version=13,
    )


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    ROOT = './runs/'
    parser.add_argument('--weights', type=str, default=ROOT + 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default=ROOT + 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT + 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--backend', type=str, choices=['tensorrt', 'openvino'])
    parser.add_argument('--log-dir', type=str, help='Path to dir for quantization log')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for distillation')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
