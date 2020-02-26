CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=2 allennlp train configs/HSCRF_softDictionary.conll2003.config -s dump_directory/ --include-package models --recover
