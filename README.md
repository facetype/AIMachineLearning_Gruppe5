# AIMachineLearning_Gruppe5

Automatic classification of 12-lead ECG recordings into diagnostic categories, using the
[PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/). See `oppgave.md` for the full
assignment and `diagnoser.md` for the diagnostic classes.

## Project structure

```
data/
  raw/         PTB-XL dataset (not in git, download it yourself)
  processed/   cached preprocessed data (not in git, generated)
src/           shared code imported by scripts and notebooks
scripts/       runnable entry points (data preparation, training)
notebooks/     data exploration and plots
results/       figures and metrics for the report
models/        saved trained models
```

## Setup

1. Download PTB-XL v1.0.3 from the link above and extract it into `data/raw/`, so you get
   `data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`.
2. Create a virtual environment and install the dependencies:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas wfdb matplotlib torch scikit-learn
   ```

