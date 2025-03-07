# debiased-BN
This repository contains materials and code related to MICCAI 2025

## Project Structure

- **config.py**: Configuration and hyperparameters.
- **utils.py**: Utility functions.
- **data_processing.py**: Data loading, splitting, and normalization.
- **model.py**: Model definitions and helper functions.
- **dfr.py**: Functions for DFR hyperparameter tuning and evaluation.
- **debiased_bn.py**: Class for Debaised batch normalization.
- **training.py**: Training and evaluation routines.
- **main.py**: Main script to run the experiments.
- **requirements.txt**: Python package dependencies.


## Debiased Batch Normalization

Easily implement debiased batch normalization by simply replacing the standard PyTorch BatchNorm with our custom `DebiasedBatchNorm` class. For example:

```python
# Replace this:
self.batch_norm1 = nn.BatchNorm1d(embed_dim)

# With this:
from debiased_bn import DebiasedBatchNorm
self.batch_norm1 = DebiasedBatchNorm(embed_dim, sub_batch_num, sub_batch_size) # batch size = sub_batch_num * sub_batch_size # 64, 8, 8
