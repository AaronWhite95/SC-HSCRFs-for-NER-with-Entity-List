import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


model = torch.load("./data/best.th")
#print(model["word_embeddings.token_embedder_tokens.weight"]),exit(0)
#model = torch.load("/home/xfbai/acl19_subtagger-master/download/state.th")
for ky,val in model.items():
    print(ky, val.size())