local token_emb_dim = 20;

{
  "dataset_reader": {
    "type": "smartdata-ner-reader",
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
    "type": "crf_tagger",
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
      "hidden_size": 20,
      "num_layers": 1,
      "bidirectional": true,
    },
    "label_encoding": "BIO",
    "calculate_span_f1": true,
    "dropout": 0.0,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 10,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 100,
    "grad_clipping": 5.0,
    "cuda_device": -1,
  },
}