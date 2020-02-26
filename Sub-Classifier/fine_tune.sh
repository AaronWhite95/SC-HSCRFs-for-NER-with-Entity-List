CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=0 allennlp fine-tune -m ./fine_tune/model.tar.gz -c configs/fine_tune.jsonnet -s new_fine_tune/ --include-package library
