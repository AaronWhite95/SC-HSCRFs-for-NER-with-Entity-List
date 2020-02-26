CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=3 allennlp train configs/bert.jsonnet -s dump_directory/ --include-package library --recover
