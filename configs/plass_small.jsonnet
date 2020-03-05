local token_emb_dim = 50;
local entity_emb_dim = 30;
local encoder_hidden_dim = 50;

{
  "dataset_reader": {
    "type": "smartdata-eventx-reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "data/smartdata-sdw-events/train.jsonl",
  "validation_data_path": "data/smartdata-sdw-events/dev.jsonl",
  "model": {
    "type": "daystream-eventx-model",
    "hidden_dim": encoder_hidden_dim,
//    "loss_weight": 5.0,
//    "trigger_gamma": 3,
//    "role_gamma": 3,
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
    "batch_size": 16,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 10,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "validation_metric": "+role_f1",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}