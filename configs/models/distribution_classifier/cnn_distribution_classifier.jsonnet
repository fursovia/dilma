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
    "type": "text_classification_json",
    // DO NOT CHANGE token_indexers
    "token_indexers": TOKEN_INDEXER,
    // DO NOT CHANGE tokenizer
    "tokenizer": {
      "type": "just_spaces"
    },
    "skip_label_indexing": true,
    "lazy": false
  },
  "train_data_path": std.extVar("CLS_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("CLS_VALID_DATA_PATH"),
  // Make sure you load vocab from LM
  "vocabulary": {
    "type": "from_files",
    "directory": std.extVar("LM_VOCAB_PATH")
  },
  "model": {
    "type": "distribution_classifier",
    "masked_lm": {
        "type": "from_archive",
        "archive_file": std.extVar("LM_ARCHIVE_PATH")
    },
    "seq2vec_encoder": {
      "type": "distribution_cnn",
      "num_filters": 8,
      "ngram_filter_sizes": [
        3,
        5
      ]
    },
    "num_labels": std.parseInt(std.extVar("CLS_NUM_CLASSES"))
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
