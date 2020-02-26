from typing import Dict
from overrides import overrides
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_final_encoder_states

import numpy as np
import torch
#torch.set_printoptions(threshold = 1e6)
import torch.nn as nn
import torch.nn.functional as F

# Model in AllenNLP represents a model that is trained.
@Model.register("fine_tune_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 pretrained_model_path: str,
                 positive_label: int = 1) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_label)

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        self.W = nn.Parameter(torch.zeros(size=(2 * encoder.get_output_dim(), 1)))
        nn.init.xavier_uniform_(self.W.data)
        self.LeakyReLU = torch.nn.LeakyReLU(0.1)

        '''
        self.W.data = self.load_weight(
            pretrained_model_path,
            self.word_embeddings,
            self.encoder,
            self.linear
        )
        print("W.requires_grad:",self.W.requires_grad)
        self.W = self.load_weight(
            pretrained_model_path,
            self.word_embeddings,
            self.encoder,
            self.linear
        )
        self.W
        '''
    
    def load_weight(
        self,
        pretrained_model_path,
        word_embeddings,
        encoder,
        linear
    ):
        pretrained_model_state = torch.load(pretrained_model_path)
        
        attention_W = pretrained_model_state["W"]
        attention_W.requires_grad = True

        
        word_embeddings_statedict = {"token_embedder_tokens.weight": pretrained_model_state["word_embeddings.token_embedder_tokens.weight"]}
        #tmp = pretrained_model_state["word_embeddings.token_embedder_tokens.weight"]
        #word_embeddings_statedict["token_embedder_tokens.weight"] = torch.cat([tmp, tmp[0:1].expand(self.vocab.get_vocab_size('tokens') - tmp.size(0), -1)], dim=0)


        word_embeddings.load_state_dict(word_embeddings_statedict)
        word_embeddings.eval()
        for param in word_embeddings.parameters():
            param.requires_grad = True
        

        
        encoder_statedict = {ky[len("encoder")+1:]:val for ky, val in pretrained_model_state.items() if ky.startswith("encoder")}
        encoder.load_state_dict(encoder_statedict)
        encoder.eval()

        for param in encoder.parameters():
            param.requires_grad = True

        linear_dict = {ky[len("linear")+1:]: val for ky, val in pretrained_model_state.items() if ky.startswith("linear")}
        linear.load_state_dict(linear_dict)
        linear.eval()

        for param in linear.parameters():
            param.requires_grad = True

        return attention_W


    def attention(self, lstm_output, final_state):
        #print(lstm_output),print(lstm_output.size()),print(final_state),print(final_state.size())
        new_final_state = torch.unsqueeze(final_state,1).expand(final_state.size()[0], lstm_output.size()[1], final_state.size()[-1])
        merge = torch.cat([lstm_output, new_final_state], -1)
        #print(merge),print(merge.size()),exit(0)
        #e = self.LeakyReLU(torch.matmul(merge, self.W)).squeeze(-1)
        e = self.LeakyReLU(torch.matmul(merge, self.W)).squeeze(-1)
        attention = F.softmax(e, dim=1).unsqueeze(1)
        attention_out = torch.matmul(attention, lstm_output).squeeze(1)
        #print(attention),print(attention_out),print(attention_out.size()),exit(0)
        return attention_out




    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        #print(tokens)
        mask = get_text_field_mask(tokens)

        
        '''
        for param in self.word_embeddings.parameters():
            #print(param.size()),exit(0)
            with open("embed.npy", "rb") as nf:
                p1 = np.load(nf)
            print(param.equal(torch.from_numpy(p1).cpu()))
        
            #np.save("embed.npy", param.detach().numpy())
            model = torch.load("./new/best.th")
            param1 = model["word_embeddings.token_embedder_tokens.weight"]
            print(param1.size())
            #print(param.equal(param1.cpu()))
            print(param.equal(param1.cpu()))
        for param in self.encoder.parameters():
            print(param)
        '''

        #embed = torch.from_numpy(np.load("embed.npy"))
        
        #state_dict = {"token_embedder_tokens.weight": embed}
        #self.word_embeddings.load_state_dict(state_dict)

        # Forward pass
        #print(tokens)
        embeddings = self.word_embeddings(tokens)

        #print("embed:",embeddings)
        encoder_out = self.encoder(embeddings, mask)
        #print(encoder_out)

        
        #print(mask, mask.size()),exit(0)
        last_state = get_final_encoder_states(encoder_out, mask)
        attention_out = self.attention(encoder_out, last_state)

        logits = F.softmax(self.linear(attention_out), dim=-1)
        print(logits)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            #print(logits,label)
            output["loss"] = self.loss_function(logits, label)

        return output
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}
