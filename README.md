<div align="center">
<p>
   <a align="left" href="https://enot.ai" target="_blank">
   <img width="850" src="splash.jpg"></a>
</p>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset.  

This repo shows how to accelerate YOLOv5 model by pruning and quantization.
</p>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.  
See the [ENOT Docs](https://enot-autodl.rtd.enot.ai/en/latest/) for documentation on pruning and quantization.  

## <div align="center">Quick Start Examples</div>

Clone repo and install [requirements.txt](https://github.com/ENOT-AutoDL/yoqo/blob/master/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch==1.13.1**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

Note: We recommend to install **enot-autodl==3.4.6** and **enot-lite==0.8.1** following these instructions:  
[**enot-autodl installation instruction**](https://enot-autodl.rtd.enot.ai/en/latest/installation_guide.html)   
[**enot-lite installation instruction**](https://enot-lite.rtd.enot.ai/en/latest/installation_guide.html)  

```bash
mkdir -p $HOME/.hasplm
echo -e 'broadcastsearch = 0\nserveraddr = 65.109.162.71\ndisable_IPv6 = 0' > $HOME/.hasplm/hasp_26970.ini
pip install enot-autodl==3.4.6
wget -O - https://raw.githubusercontent.com/ENOT-AutoDL/ONNX-Runtime-with-TensorRT-and-OpenVINO/master/install.sh | bash
pip install enot-lite==0.8.1
```

## <div align="center">Demo</div>
There is [**demo notebook**](https://github.com/ENOT-AutoDL/yoqo/blob/master/demo.ipynb) which shows how to prune and quantize YOLOv5 model.


## <div align="center">Contact</div>

**enot@enot.ai**
