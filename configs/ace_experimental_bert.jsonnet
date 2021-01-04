local bert_model = "bert-base-uncased";
local token_emb_dim = 768;
local entity_emb_dim = 50;
local encoder_hidden_dim = 300;

{
  "random_seed": std.parseInt(std.extVar("SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "dataset_reader": {
    "type": "experimental-ace2005-reader",
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
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("DEV_PATH"),
  "model": {
    "type": "experimental-ace-model",
    "hidden_dim": encoder_hidden_dim,
    "loss_weight": 5.0,
//    "trigger_gamma": 0.5,
//    "role_gamma": 0.5,
    "positive_class_weight": 5.0,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
        "ner_tokens": ["ner_tokens"],
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
      "embedding_dim": entity_emb_dim,
      "trainable": true,
      "vocab_namespace": 'entity_tags',
    },
    "trigger_embedder": {
      "type": "embedding",
      "embedding_dim": 30,
      "trainable": true,
      "vocab_namespace": 'trigger_labels',
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
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]],
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.0022260678803619886,
    },
    "num_serialized_models_to_keep": 1,
    "patience": 20,
    "validation_metric": "+role_f1",
    "num_epochs": 100,
    "grad_clipping": 5.0,
    "cuda_device": 0,
  },
}