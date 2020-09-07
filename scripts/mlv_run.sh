# Run 5 random repeats

#
# edit these variables before running script
export CUDA_DEVICE=0
export NUM_EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=0.0022260678803619886
DATA_DIR=daystream_corpus
export TRAIN_PATH=data/"$DATA_DIR"/daystream_mlv_snorkeled.jsonl
export DEV_PATH=data/"$DATA_DIR"/dev/dev_with_events_and_defaults.jsonl
CONFIG_FILE=configs/snorkel_bert.jsonnet

ITER=1
for RANDOM_SEED in 54360 44184 20423 80520 27916; do

	SEED=$RANDOM_SEED
	PYTORCH_SEED=`expr $RANDOM_SEED / 10`
	NUMPY_SEED=`expr $PYTORCH_SEED / 10`
	export SEED=$SEED
	export PYTORCH_SEED=$PYTORCH_SEED
	export NUMPY_SEED=$NUMPY_SEED

	echo Run ${ITER} with seed ${RANDOM_SEED}

  OUTPUT_DIR=data/runs/random_repeats/run_"$ITER"/snorkel_bert_mlv_daystream

  allennlp train --include-package eventx $CONFIG_FILE -s $OUTPUT_DIR -f

	ITER=$(expr $ITER + 1)
done
