# DiffLoad: Uncertainty Quantification in Load Forecasting with the Diffusion Model

## Intro

This work proposes a diffusion-based Seq2seq structure to estimate epistemic uncertainty and employs the robust additive Cauchy distribution to estimate aleatoric uncertainty, which not only ensures the accuracy of load forecasting but also demonstrates the ability to separate the two types of uncertainties for different levels of loads, giving more information for the downstream decision makers.

Codes for Paper "DiffLoad: Uncertainty Quantification in Load Forecasting with the Diffusion Model".

Authors: Zhixian Wang, Qingsong Wen, Chaoli Zhang, Liang Sun, and Yi Wang.

## Prerequisites
- Python 
- Conda

### Initial packages include
  - python=3.8.18
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - plotly
  - statsmodels
  - xlrd
  - jupyterlab
  - nodejs
  - mypy
  - pytorch
  - scipy
  - blitz


## Experiments 

We have provided corresponding codes for all uncertainty estimation methods mentioned in the paper, such as the Bayesian neural network. Please run __train.py__ and __test.py__ respectively in the corresponding folders to reproduce the relevant results.

```
cd seq2seq_diffusion
python train.py
python test.py
```
