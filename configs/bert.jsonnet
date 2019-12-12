local bert_model = "bert-base-uncased";
local encoder_hidden_dim = 200;
local SEED = 0;

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "ace2005-reader",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
    },
  },
  "train_data_path": "data/ace05/train.json",
  "validation_data_path": "data/ace05/dev.json",
  "model": {
    "type": "eventx-model",
    "hidden_dim": encoder_hidden_dim,
    "loss_weight": 10,
    "trigger_gamma": 3.0,
    "role_gamma": 3.0,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
        },
      },
    },
    "entity_embedder": {
      "type": "embedding",
      "embedding_dim": 30,
      "trainable": true,
      "vocab_namespace": 'entity_labels',
    },
    "trigger_embedder": {
      "type": "embedding",
      "embedding_dim": 30,
      "trainable": true,
      "vocab_namespace": 'trigger_labels',
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": 0.3,
    },
    "span_extractor": {
      "type": "bidirectional_endpoint",
      "input_dim": 2 * encoder_hidden_dim,
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 10,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "validation_metric": "-loss",
    "num_epochs": 100,
//    "grad_clipping": 5.0,
  },
}