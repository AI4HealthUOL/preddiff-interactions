# *PredDiff*: Explanations and Interactions from Conditional Expectations


This repository provides all resources to reproduce the paper:
*PredDiff*: Explanations and Interactions from Conditional Expectations

We provide ready-to-run jupyter notebooks, which apply *PredDiff* on different datasets
* **Synthetic regression** (`synthetic_dataset.ipynb`): (Interaction) relevances for a regressor on the synthetic dataset discussed in the paper
* **MNIST** (`mnist.ipynb`): (Interaction) relevances for a classifier trained on MNIST seen as a tabular dataset
* **NHANES** (`nhanes.ipynb`): (Interaction) relevances for a classifier trained on the NHANES (mortality regression) dataset

# Requirements
Install dependencies from `pred_diff.yml` by running `conda env create -f pred_diff.yml` and activate the environment via `conda activate pred_diff`
