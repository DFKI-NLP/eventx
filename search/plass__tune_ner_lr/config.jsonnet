local learning_rate = std.extVar("LEARNING_RATE");
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "smartdata-ner-reader",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "do_lowercase": false,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
    },
  },
  "train_data_path": "/home/huebner/code/eventx/data/smartdata-sdw-events/train.jsonl",
  "validation_data_path": "/home/huebner/code/eventx/data/smartdata-sdw-events/dev.jsonl",
  "test_data_path": "/home/huebner/code/eventx/data/smartdata-sdw-events/test.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
        "ner_tokens": ["ner_tokens"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased.tar.gz",
        },
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 300,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": 0.2,
    },
    "label_encoding": "BIO",
    "calculate_span_f1": true,
    "dropout": 0.2,
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]],
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": learning_rate,
    },
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 100,
    "grad_clipping": 5.0,
    "cuda_device": 0,
  },
}