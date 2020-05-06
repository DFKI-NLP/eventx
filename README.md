# Event extraction implementation

Jointly extracts multiple events: event detection and argument extraction for multiple events in one pass
Loosely based on the paper "Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation" (EMNLP 2018)

## Getting started on ACE 2005

### Create and activate a conda environment
    conda create -n eventx python=3.7
    conda activate eventx

### Install package requirements
    pip install -r requirements.txt

### Preprocess the ACE 2005 corpus
 * The ACE 2005 corpus can be received from the LDC (https://catalog.ldc.upenn.edu/LDC2006T06).
 * To preprocess the corpus, follow the instructions in the following repository: https://github.com/marchbnr/ace2005-preprocessing
 * Put the files into the following directory: `data/ace05`

