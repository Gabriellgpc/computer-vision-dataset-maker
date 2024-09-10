# Computer Vision Dataset Maker

You can find a technical article related to this project here [The Power of Florence-2 with OpenVINO & FiftyOne: Real-World Applications in Image Analysis](https://medium.com/@condadoslgpc/the-power-of-florence-2-with-openvino-fiftyone-real-world-applications-in-image-analysis-b931fd8adb44)


## Conda environment

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