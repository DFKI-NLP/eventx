# Event extraction implementation

Jointly extracts multiple events: event detection and argument extraction for multiple events in one pass
Loosely based on the paper "Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation" (EMNLP 2018)

## Getting started

### Create and activate a conda environment
    conda create -n eventx python=3.7
    conda activate eventx

### Install package requirements
    pip install -r requirements.txt

### Preprocess the data
Download the preprocssed SD4M+Daystream data from 
[https://dfkide-my.sharepoint.com/:u:/g/personal/lehe02_dfki_de/Ecl1cWZlZVZNg844cZPUdOgB3MGjIapyeCGyDfpx_kylUw](https://dfkide-my.sharepoint.com/:u:/g/personal/lehe02_dfki_de/Ecl1cWZlZVZNg844cZPUdOgB3MGjIapyeCGyDfpx_kylUw&Web=0&Download=1).
And extract it into the data directory.

### Train the models
In order to train the different models you can use the [scripts](scripts) in the repository.
You may need to adjust the configuration file, the training & development data paths and the save paths.
- [train_eventx_snorkel.sh](scripts/train_eventx_snorkel.sh): Train a single model.
- [random_repeats.sh](scripts/random_repeats.sh): Random repeats with different seeds.
- [increasing_train_data.sh](scripts/increasing_train_data.sh): Random repeats with different seeds for increasing training data.
- [mlv_run.sh](scripts/mlv_run.sh): Random repeats with different seeds for Daystream data labeled with Majority Label Voter. 

E.g. to train a single model:
```
./scripts/train_eventx_snorkel.sh data/training_run_1
```

In order to recreate the models in our main experiments, you need to run [random_repeats.sh](scripts/random_repeats.sh).

For our other experiments, you need to run [increasing_train_data.sh](scripts/increasing_train_data.sh) (Increasing Daystream
training data) and 
[mlv_run.sh](scripts/mlv_run.sh) (Majority Label Voter).

For convenience we provide the trained event extraction model for the setup with the merged training set (manual annotation + annotation with the Snorkel approach):  [https://dfkide-my.sharepoint.com/:u:/g/personal/lehe02_dfki_de/ESEeQWkkzRlKpPl95e8HSeEBJdl1W9N2e4d246onXQmJiQ](https://dfkide-my.sharepoint.com/:u:/g/personal/lehe02_dfki_de/ESEeQWkkzRlKpPl95e8HSeEBJdl1W9N2e4d246onXQmJiQ?Web=0&Download=1)
