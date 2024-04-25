<div align="center">

# Sparse Feature Circuit Development (wip)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Code to train [sparse autoencoders](https://transformer-circuits.pub/2023/monosemantic-features), use those to identify 
sparse feature circuits and then investigate the development of these during training.

### TODO
#### Part 1 - Sparse Autoencoders
- [ ] Implement training code for sparse autoencoders
  - [x] Implement activation buffer
  - [ ] Implement sparse autoencoder
  - [ ] Implement pytorch lightning training code
- [ ] Train sparse autoencoders for 1L transformer and verify that the results are similar to e.g. [this](https://github.com/neelnanda-io/1L-Sparse-Autoencoder)
- [ ] Determine best open model to focus on (Pythia 14M?)
- [ ] Train sparse autoencoders for all relevant layers for selected model
- [ ] Investigate features for this model
- [ ] Train for several (number dependent on the time it takes to train) model checkpoints from different points during training
- [ ] Investigate ways to match up features throughout training so we can see how they develop

#### Part 2 - Sparse Feature Circuits
- [ ] Implement code to identify sparse feature circuits
- [ ] Run this for all layers and several checkpoints from different points during training
- [ ] Investigate ways to match up circuits throughout training so we can see how they develop


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/RoganInglis/sparse-feature-circuit-development
cd sparse-feature-circuit-development

# [OPTIONAL] create conda environment
conda create -n venv python=3.9
conda activate venv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/RoganInglis/sparse-feature-circuit-development
cd sparse-feature-circuit-development

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
