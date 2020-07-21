# Train model for each sample size

#
# edit these variables before running script
export CUDA_DEVICE=0
export NUM_EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=0.0022260678803619886
DATA_DIR=pipeline_run
export DEV_PATH=data/"$DATA_DIR"/dev/dev_with_events_and_defaults.jsonl
CONFIG_FILE=configs/snorkel_bert.jsonnet

SEED=13370
PYTORCH_SEED=`expr $RANDOM_SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

for SAMPLE_FRACTION in 50 60 70 80 90 100; do
  SAMPLE=daystream"$SAMPLE_FRACTION"_snorkeled
  TRAIN_PATH=data/"$DATA_DIR"/"$SAMPLE".jsonl
  export TRAIN_PATH
  OUTPUT_DIR=data/runs/"$SAMPLE"

  allennlp train --include-package eventx $CONFIG_FILE -s $OUTPUT_DIR -f
done
