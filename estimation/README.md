# Causal Effect Identification in lvLiNGAM from High-Order Cumulants

This project implements the code to reproduce the experiments in the paper ``Causal Effect Identification in lvLiNGAM from High-Order Cumulants''.

## Directory Structure
- **Figures/** Contains the figures present in the paper.
- **Results/** Contains that data to reproduce the figures in the paper.
- **helpers/** Contains all the function to run the algorithms presented in the paper
- **other_moodels/** Contains the functions to run the mehtods we compared our algoritithm with.
- **iv_experiments.ipynb** The code to reproduce the exeriments on the IV graph.
- **proxy_experiments.ipynb** The code to reproduce the experiments on the Proxy variable graph.
- **plots.ipynb** The code to create the figures.



Recreate environment:
```
conda env create -f requirements.yaml
conda activate lvlingam
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=lvlingam
```

