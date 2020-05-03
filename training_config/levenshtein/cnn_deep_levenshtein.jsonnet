local TOKEN_INDEXER = {
    "tokens": {
        "type": "single_id",
        "start_tokens": [
          "<START>"
        ],
        "end_tokens": [
          "<END>"
        ],
        // should be set to the maximum value of `ngram_filter_sizes`
        "token_min_padding_length": 5
      }
};

{
  "dataset_reader": {
    "type": "deep_levenshtein",
    // DO NOT CHANGE token_indexers
    "token_indexers": TOKEN_INDEXER,
    // DO NOT CHANGE tokenizer
    "tokenizer": {
      "type": "just_spaces"
    },
    "lazy": true
  },
  "train_data_path": std.extVar("DL_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("DL_VALID_DATA_PATH"),
  // Make sure you load vocab from LM
  "vocabulary": {
    "type": "from_files",
    "directory": std.extVar("LM_VOCAB_PATH")
  },
  "model": {
    "type": "deep_levenshtein",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "trainable": true
        }
      }
    },
    "seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": 100,
      "num_filters": 8,
      "ngram_filter_sizes": [
        3,
        5
      ]
    }
  },
  "data_loader": {
    "batch_size": 64
  },
  "distributed": {
    "cuda_devices": [
      0,
      2,
      3
    ]
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3
  }
}
