local embedding_dim = 300;
local hidden_dim = 128;
local batch_size = 32;
local num_epochs = 3;
local patience = 5;
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
  "train_data_path": "data/train.txt",
  "validation_data_path": "data/test.txt",
  "vocabulary": {
      "directory_path": "/home/xfbai/CON_NER/subclassifier_NER/vocabulary/SoftDictHSCRF_vocabulary",
      "extend": true
  },
  "model": {
    "type": "lstm_classifier",

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

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "batch_first": "True"
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
