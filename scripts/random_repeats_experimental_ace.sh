# Run 5 random repeats

#
# edit these variables before running script
export CUDA_DEVICE=0
export NUM_EPOCHS=100
export BATCH_SIZE=32
CONFIG_FILE=configs/ace_experimental_bert.jsonnet
export TRAIN_PATH=data/ace05/train.json
export DEV_PATH=data/ace05/dev.json

ITER=1
for RANDOM_SEED in 54360 44184 20423 80520 27916; do

	SEED=$RANDOM_SEED
	PYTORCH_SEED=`expr $RANDOM_SEED / 10`
	NUMPY_SEED=`expr $PYTORCH_SEED / 10`
	export SEED=$SEED
	export PYTORCH_SEED=$PYTORCH_SEED
	export NUMPY_SEED=$NUMPY_SEED

	echo Run ${ITER} with seed ${RANDOM_SEED}
	OUTPUT_DIR=data/runs/random_repeats_ace_lr/run_"$ITER"

	allennlp train --include-package eventx $CONFIG_FILE -s $OUTPUT_DIR -f

	ITER=$(expr $ITER + 1)
done
