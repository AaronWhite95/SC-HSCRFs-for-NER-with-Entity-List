local embedding_dim = 300;
local hidden_dim = 128;
local batch_size = 128;
local num_epochs = 10;
local patience = 10;
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "bertclassification",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    }
  },
  "train_data_path": "data/new_train.txt",
  "validation_data_path": "data/new_train.txt",
  "vocabulary": {
      "directory_path": "/home/xfbai/SentimentClassifier-master/vocabulary/SoftDictHSCRF_vocabulary/",
      "extend": true
  },
  "model": {
    "type": "fine_tune_classifier",

    "word_embeddings": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "./data/sgns.baidubaike.bigram-char",
          "trainable": true
        }
      }
    },
    "pretrained_model_path": "./new/best.th",
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "dropout": 0.5,
      "bidirectional": true
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": batch_size,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
            "type": "adam"
    },
    "num_epochs": num_epochs,
    "patience": patience,
    "cuda_device": cuda_device
  }
}
