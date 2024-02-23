# DriveLLaVA

[![codecov](https://codecov.io/gh/AdityaNG/DriveLLaVA/branch/main/graph/badge.svg?token=DriveLLaVA_token_here)](https://codecov.io/gh/AdityaNG/DriveLLaVA)
[![CI](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml)

Training LLaVA on the CommaVQ dataset to produce a tokenized trajectory

## Install it from PyPI

```bash
pip install drivellava
```

## Usage

```py
from drivellava import BaseClass
from drivellava import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m drivellava
#or
$ drivellava
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Dataset setup

```python3
python3 -m drivellava.scripts.generate_commavq_images
```

## TODO

- Training script
  - Select what layers to train
  - Measure memory requirements
- Dataset
  - Generate images from CommaVQ
  - Quantize the trajectory
  - Visualize the trajectory on the image
  - Generate JSON dataset

## References

- Fine-tuning LLaVA: https://ubiai.tools/how-to-fine-tune-llava-on-your-custom-dataset/