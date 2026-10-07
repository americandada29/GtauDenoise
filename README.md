# GtauDenoise

A Python package for denoising imaginary-time Green's functions using auxiliary Green's functions.

## Installation
pull this repository, then execute
pip install -e .

### Requirements

GtauDenoise requires Python 3.10 or newer and is designed to work with the following package versions:

| Package | Supported versions |
| --- | --- |
| Python | `>=3.10` |
| PyTorch | `>=2.0,<3.0` |
| NumPy | `>=1.24,<3.0` |
| SciPy | `>=1.10,<2.0` |
| Matplotlib | `>=3.7,<4.0` |
| tqdm | `>=4.60,<5.0` |
| TRIQS | `>=3.0,<4.0` |

### Example script
In the scripts folder, vbhubbard_denoise_accel.py gives an example of how to run the denoising code. One can either generate data for normalization on the fly (done within the script) or if a model has already been trained, then the training data from that can be reused as normalization data. Weights for our $\beta$=200 model are provided. Changing $\beta$ requires retraining the model.
