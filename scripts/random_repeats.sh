# Run 5 random repeats

#
# edit these variables before running script
export CUDA_DEVICE=0
export NUM_EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=0.0022260678803619886
DATA_DIR=daystream_corpus
export DEV_PATH=data/"$DATA_DIR"/dev/dev_with_events_and_defaults.jsonl
CONFIG_FILE=configs/snorkel_bert.jsonnet
OLDIFS=$IFS
ITER=1
for RANDOM_SEED in 54360 44184 20423 80520 27916; do

	SEED=$RANDOM_SEED
	PYTORCH_SEED=`expr $RANDOM_SEED / 10`
	NUMPY_SEED=`expr $PYTORCH_SEED / 10`
	export SEED=$SEED
	export PYTORCH_SEED=$PYTORCH_SEED
	export NUMPY_SEED=$NUMPY_SEED

	echo Run ${ITER} with seed ${RANDOM_SEED}
  IFS=','
	for CONFIG in daystream_snorkeled,snorkel_bert_daystream sd4m_gold,snorkel_bert_gold snorkeled_gold_merge,snorkel_bert_merged; do set -- $CONFIG
	  TRAIN_PATH=data/"$DATA_DIR"/run_"$ITER"/"$1".jsonl
    if [ "$CONFIG" == "sd4m_gold" ]; then
	    TRAIN_PATH=data/"$DATA_DIR"/train/train_with_events_and_defaults.jsonl
	  fi
	  export TRAIN_PATH
		OUTPUT_DIR=data/runs/random_repeats/run_"$ITER"/"$2"

		allennlp train --include-package eventx $CONFIG_FILE -s $OUTPUT_DIR -f
	done

	ITER=$(expr $ITER + 1)
done
IFS=$OLDIF
