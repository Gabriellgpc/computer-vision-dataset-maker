# Computer Vision Dataset Maker

```bash
conda create -n cvd-maker python=3.11 -y
conda activate cvd-maker
pip install -r requirements.txt
```

# TODO
- [x] Make fiftyone dataset
- [x] Add Object detection predictions from OVFlorence to fiftyone dataset
- [x] Add Captioning
- [x] Add Florence-2 embeddings and visualize it on 2D
- [] predictions/labels from florence-2 to yolo format
- [] Quantization to int8 and make it avaible on Huggingface models
- [] Add segmentation label using EfficientSAM or FastSAM, OpenVINO format

# Source Material and useful links

- [OpenVINO-Demo-Florence2](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/florence2/florence2.ipynb)

- [pexe-downloader](https://github.com/Gabriellgpc/pexel-downloader)