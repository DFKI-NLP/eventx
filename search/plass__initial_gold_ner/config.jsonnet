
local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));
local learning_rate = std.extVar("LEARNING_RATE");
local loss_weight = std.extVar("LOSS_WEIGHT");

local token_emb_dim = 50;
local entity_emb_dim = 30;
local encoder_hidden_dim = 50;

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "daystream-reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "/home/huebner/smartdata-corpus-v3.0-20200206-tmp/train/train_sdw_with_events.jsonl",
  "validation_data_path": "/home/huebner/smartdata-corpus-v3.0-20200206-tmp/dev/dev_sdw_with_events.jsonl",
  "model": {
    "type": "daystream-eventx-model",
    "hidden_dim": encoder_hidden_dim,
    "loss_weight": loss_weight,
//    "trigger_gamma": 0.5,
//    "role_gamma": 0.5,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
//          "pretrained_file": "/home/marc/Downloads/vectors.txt",
          "trainable": true,
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
      "input_size": token_emb_dim + entity_emb_dim,
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
    "type": "basic",
    "batch_size": 20,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": learning_rate,
    },
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "-loss",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}