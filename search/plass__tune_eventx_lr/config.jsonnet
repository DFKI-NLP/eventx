local learning_rate = std.extVar("LEARNING_RATE");
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));
local entity_emb_dim = 50;
local encoder_hidden_dim = 300;

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "smartdata-eventx-reader",
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
    "type": "daystream-eventx-model",
    "hidden_dim": encoder_hidden_dim,
//    "loss_weight": 5.0,
//    "trigger_gamma": 3,
//    "role_gamma": 3,
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
    "entity_embedder": {
      "type": "embedding",
      "embedding_dim": entity_emb_dim,
      "trainable": true,
      "vocab_namespace": 'entity_tags',
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768 + entity_emb_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 2,
      "bidirectional": true,
    },
    "span_extractor": {
      "type": "bidirectional_endpoint",
      "input_dim": 2 * encoder_hidden_dim,
    }
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
    "validation_metric": "+role_f1",
    "num_epochs": 100,
    "grad_clipping": 5.0,
    "cuda_device": 0,
  },
}
