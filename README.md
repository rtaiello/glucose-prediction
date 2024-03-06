# Glucose Prediction

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.8-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Glucose Prediction using [Replace-BG Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5864100/pdf/dc162482.pdf) <br>
- Preprocessing insperid by [Long-Term Prediction of Blood Glucose
 Levels in Type 1 Diabetes Using a CNNLSTM-Based Deep Neural Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10658677/pdf/10.1177_19322968221092785.pdf)
- Code insperid by [GluPred Github Repository](https://github.com/r-cui/GluPred)

## Installation

```bash
pip install git+ssh://git@github.com/rtaiello/glucose-prediction.git
```


## Quickstart

- Download `REPLACE-BG-Dataset.zip` from this [link](https://public.jaeb.org/datasets/diabetes);
- `unzip` it and move `HDeviceBolus.txt`, `HDeviceCGM.txt` and `HDeviceWizard.txt` to `data/original` folder;
- Run preprocessing notebook, [preprocessing.ipynb](https://github.com/rtaiello/glucose-prediction/blob/main/src/glucose_prediction/preprocessing.ipynb)

## Development installation

Setup the development environment:

```bash
git clone git@github.com:rtaiello/glucose-prediction.git
cd glucose-prediction
conda env create -f env.yaml
conda activate glucose-prediction
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
