local TOKEN_INDEXER = {
    "tokens": {
        "type": "single_id"
      }
};

{
  "dataset_reader": {
    // this is not a mistake
    "type": "simple_language_modeling_fixed",
    // DO NOT CHANGE token_indexers
    "token_indexers": TOKEN_INDEXER,
    // DO NOT CHANGE tokenizer
    "tokenizer": {
      "type": "just_spaces"
    },
    // must be lower than positional_embedding_size
    "max_sequence_length": 150,
    "lazy": false
  },
  "train_data_path": std.extVar("LM_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("LM_VALID_DATA_PATH"),
  "vocabulary": {
    "tokens_to_add": {
      "max_vocab_size": {
        "tokens": 50000
      },
      "tokens": [
        "@@MASK@@",
        "<START>",
        "<END>"
      ]
    },
  },
  "model": {
    "type": "language_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128,
          "trainable": true
        }
      }
    },
    "contextualizer": {
      "type": "lstm",
      "input_size": 128,
      "hidden_size": 256,
      "num_layers": 1,
    }
  },
  "distributed": {
    "master_port": 29599,
    "cuda_devices": [
      1,
      2,
      3
    ]
  },
  "data_loader": {
    "batch_size": 256
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3
  }
}
