{
  "dataset_reader": {
    // this is not a mistake
    "type": "text_classification_json",
    // DO NOT CHANGE token_indexers
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "start_tokens": [
          "<START>"
        ],
        "end_tokens": [
          "<END>"
        ]
      }
    },
    // DO NOT CHANGE tokenizer
    "tokenizer": {
      "type": "just_spaces"
    },
    "max_sequence_length": 60,
    "lazy": false
  },
  "train_data_path": std.extVar("LM_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("LM_VALID_DATA_PATH"),
  "vocabulary": {
    "tokens_to_add": {
      "tokens": [
        "@@MASK@@",
        "<START>",
        "<END>"
      ]
    },
  },
  "model": {
    "type": "masked_lm",
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
      "type": "pytorch_transformer",
      "input_dim": 100,
      "num_layers": 4,
      "num_attention_heads": 4,
      "positional_encoding": "embedding",
      "positional_embedding_size": 64
    },
    "tokens_masker": {
      "type": "tokens_masker"
    }
  },
  "distributed": {
    "cuda_devices": [
      2,
      3
    ]
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3
  }
}
