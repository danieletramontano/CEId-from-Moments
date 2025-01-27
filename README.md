# Causal Effect Identification in lvLiNGAM from High-Order Cumulants

This repository contains the code to reproduce the experiments in the paper *"Causal Effect Identification in lvLiNGAM from High-Order Cumulants."*

## Directory Structure

### **Estimation/**
Python code to reproduce the estimation experiments in the paper.
- **Figures/**: Contains the figures presented in the paper.
- **Results/**: Contains the data necessary to reproduce the figures in the paper.
- **helpers/**: Includes all helper functions for running the algorithms described in the paper.
- **other_models/**: Provides the functions to run methods that our algorithm was compared against.
- **iv_experiments.ipynb**: Jupyter notebook to reproduce experiments on the Instrumental Variable (IV) graph.
- **proxy_experiments.ipynb**: Jupyter notebook to reproduce experiments on the Proxy Variable graph.
- **plots.ipynb**: Jupyter notebook to generate the figures from the experimental results.

### **Macaulay2/**
Macaulay2 (M2) code for theoretical proofs and additional verifications in the paper.
- **NonGaussianIdentifiability.m2**: Code for testing identifiability from second- and third-order cumulants of a parameter in a specified graph.
- **proxy_polynomial.m2**: Degree-2 polynomial proving the non-identifiability of the causal effect in the Proxy Variable graph with one latent confounder.

## Recreating the Environment

To set up the environment and reproduce the experiments, follow these steps:

```bash
cd Estimation
conda env create -f requirements.yaml
conda activate lvlingam
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=lvlingam
```
## License
See `LICENSE` for more information.

<!-- Authors -->
## Authors
Anonymous.
