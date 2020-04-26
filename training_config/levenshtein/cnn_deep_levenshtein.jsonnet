{
  "dataset_reader": {
    "type": "deep_levenshtein",
    "lazy": false
  },
  "train_data_path": "data/json_ins_dl/train.json",
  "validation_data_path": "data/json_ins_dl/test.json",

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
        "num_filters": 32
     }
  },
  "data_loader": {
      "batch_size" : 32
  },
  "distributed": {
    "cuda_devices": [2, 3]
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3
  }
}
