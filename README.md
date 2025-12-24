# High-Field-Gradient Dielectrophoretic Platform Enables Small-Molecule Metabolite Redistribution in Biofluids
This repository contains model implementations and other relevant code for the paper "High-Field-Gradient Dielectrophoretic Platform Enables Small-Molecule Metabolite Redistribution in Biofluids", which has been submitted for review.

## Requirements
We provide a YAML file, which can be used to replicate the Conda environment we used to run this codebase.
```
conda env create -f environment.yml
conda activate raman
```

## Data
As this research uses datasets of clinical data, we have not uploaded said data to this repository.
The data that supports the findings of this study are available in the main article and Supporting Information file.
Any other data will be available from the corresponding author upon reasonable request.

## Models
The analysis in this paper uses three models: a modified version of ResNet to allow for 1D spectral inputs, a convolutional neural network (CNN) we call 3-CNN, and a multi-layer perceptron. Model details can be found in this repo and in the Supporting Information.