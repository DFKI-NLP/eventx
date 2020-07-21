# Run allennlp training

#
# edit these variables before running script
export CUDA_DEVICE=0
export NUM_EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=0.0022260678803619886
DATA_DIR=data/pipeline_run
CONFIG_FILE=configs/snorkel_bert.jsonnet

SEED=13370
PYTORCH_SEED=`expr $RANDOM_SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

DATA_DIR=daystream_corpus
export TRAIN_PATH=data/"$DATA_DIR"/train/train_with_events_and_defaults.jsonl
export DEV_PATH=data/"$DATA_DIR"/dev/dev_with_events_and_defaults.jsonl

allennlp train --include-package eventx $CONFIG_FILE -s "$@" -f
