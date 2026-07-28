# RadYOLO

RadYOLO is a 3D YOLO11-based deep learning model for object detection, classification, and instance segmentation in medical images. It is designed to process MRI or CT volumes efficiently on GPU and CPU-only systems. In our paper published at the 2nd MICCAI Workshop on Efficient Medical AI, we could show that RadYOLOs detection performance is on par with nnU-Net for lesion detection and coarse organ detection. At the same time it is 8-46x faster on a GPU and over 100x faster on a CPU, solving most detection tasks within a couple seconds.

RadYOLO was developed by the Fraunhofer Institute for Digital Medicine MEVIS.

If you use RadYOLO, please cite our paper:

```bibtex
@inproceedings{geissler2026radyolo,
  title={RadYOLO: Computationally Efficient 3D Object Detection and Segmentation in CT and MRI},
  author={Geissler, Kai and Müller-Groh, Laurens and Meine, Hans},
  booktitle={<enter when published>},
  year={2026},
}
```

## Quick Start: Set Up A Training

1. Install dependencies from repository root:

```bash
uv build .
```

or

```bash
pip install .
```

2. Prepare a dataset in RadYOLO format (images, labels, dataset.yaml).
3. Create data splits.
4. Optionally analyze/preprocess data and propose a model config.
5. Start training:

```bash
uv run -m radyolo train <path-to-config-yaml>
```

## Documentation

Detailed guides are available in [documentation/](documentation/README.md):

- Dataset preparation: [dataset-preparation.md](documentation/dataset-preparation.md)
- Data splits: [data-splits.md](documentation/data-splits.md)
- Analyze and preprocess data (optional): [analyze-and-preprocess.md](documentation/analyze-and-preprocess.md)
- Propose model configuration (optional): [propose-model-configuration.md](documentation/propose-model-configuration.md)
- Training quickstart: [training.md](documentation/training.md)
- Inference and evaluation: [inference-and-evaluation.md](documentation/inference-and-evaluation.md)
- Build docker image: [docker.md](documentation/docker.md)

## Disclaimer

The software is not qualified for use as a medical product or as part thereof. Provided 'as is' without specific verification or validation.

## License

RadYOLO is licensed under the [AGPL-3.0 license](LICENSE).

## Contact

Kai Geissler, kai.geissler\<at>mevis.fraunhofer.de, Fraunhofer Institute for Digital Medicine MEVIS, Germany