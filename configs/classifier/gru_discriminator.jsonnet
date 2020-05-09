{
  "dataset_reader": {
    "type": "text_classification_json",
    "token_indexers": {
        "tokens": {
            "type": "single_id"
        }
    },
    "tokenizer": {
      "type": "just_spaces"
    },
    "skip_label_indexing": true,
    "lazy": false
  },
  "train_data_path": std.extVar("DISCR_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("DISCR_VALID_DATA_PATH"),
  "model": {
    "type": "basic_classifier_one_hot_support",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "trainable": true
        }
      }
    },
    "seq2seq_encoder": {
        "type": "gru",
        "input_size": 100,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "bidirectional": true
    },
    "seq2vec_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": 256,
      "averaged": true
    },
    "dropout": 0.2,
    "num_labels": 2
  },
  "data_loader": {
    "batch_size": 64
  },
  "distributed": {
    "cuda_devices": [
      2,
      3
    ]
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3
  }
}
