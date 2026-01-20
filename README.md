# Multi-region Sparse Component Analysis (mSCA): An unsupervised method for understanding inter-region communication

## Overview
Multi-region Sparse Component Analysis (,SCA) is an unsupervised dimensionality reduction method that recovers interpretable latent factors from high dimensional neural activity from multiple brain regions. More specifically, mSCA finds latent factors that are dissociated with respect to time e.g. finding one factor related to movement preparation and another factor related to movement execution. mSCA also determines whether each latent factor is unique to or shared between brain regions. For latent factors shared between brain regions, mSCA finds the time-delay at which those factors appear in each region. This repo. contains notebooks (and example data) that demonstrate how to use mSCA. Please follow the installation instructions below to get started.

## Installation and Dependencies

This package can be installed by: 
```buildoutcfg
git clone https://github.com/AndrewUlmer/mSCA.git
cd mSCA
pip install -e .
```

Please see the example jupyter notebook **`quickstart.ipynb`** in the notebooks folder for further details on a simulated dataset. <br>
