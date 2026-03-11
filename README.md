# Glucose Prediction

<p align="center">
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Glucose Prediction using [Replace-BG Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5864100/pdf/dc162482.pdf) <br>
- Preprocessing inspired by [Long-Term Prediction of Blood Glucose Levels in Type 1 Diabetes Using a CNN-LSTM-Based Deep Neural Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10658677/pdf/10.1177_19322968221092785.pdf)
- Code inspired by [GluPred Github Repository](https://github.com/r-cui/GluPred)

## Installation

```bash
pip install git+ssh://git@github.com/rtaiello/glucose-prediction.git
```

## Quickstart

- Download `REPLACE-BG-Dataset.zip` from this [link](https://public.jaeb.org/datasets/diabetes);
- `unzip` it and move `HDeviceBolus.txt`, `HDeviceCGM.txt`, `HDeviceWizard.txt` and `HScreening.txt` to `data/original` folder;
- Run preprocessing notebook, [preprocessing.ipynb](https://github.com/rtaiello/glucose-prediction/blob/main/src/glucose_prediction/preprocessing.ipynb)

## Development installation

Setup the development environment:

```bash
git clone git@github.com:rtaiello/glucose-prediction.git
cd glucose-prediction
uv venv
source .venv/bin/activate
uv sync --extra dev
pre-commit install
```

### Training

```bash
uv run python src/glucose_prediction/run.py

# With Hydra overrides
uv run python src/glucose_prediction/run.py train.trainer.max_epochs=50 train.trainer.fast_dev_run=true

# With W&B logging
uv run python src/glucose_prediction/run.py train.logging.logger=wandb
```

### Code quality

```bash
uv run pre-commit run --all-files
uv run pytest -v
```
