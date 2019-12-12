local token_emb_dim = 2;
local encoder_hidden_dim = 2;

{
  "dataset_reader": {
    "type": "ace2005-reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "tests/fixtures/ace2005_sample_data.json",
  "validation_data_path": "tests/fixtures/ace2005_sample_data.json",
  "model": {
    "type": "eventx-model",
    "hidden_dim": encoder_hidden_dim,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "trainable": true,
        },
      },
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 1,
      "bidirectional": true,
    },
//    "span_extractor": {
//      "type": "bidirectional_endpoint",
//      "input_dim": 2 * encoder_hidden_dim,
//    }
    "span_extractor": {
      "type": "endpoint",
      "input_dim": 2 * encoder_hidden_dim,
      "combination": 'x+y',
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 5,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 1,
    "cuda_device": -1,
    "validation_metric": "-loss",
    "num_epochs": 1,
    "grad_clipping": 5.0,
  },
}