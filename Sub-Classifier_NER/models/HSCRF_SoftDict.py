from typing import Dict, Optional, List, Any
import warnings
import copy
import numpy as np

from overrides import overrides
import torch
#torch.set_printoptions(threshold = 1e6)
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward, Pruner
from allennlp.modules.conditional_random_field import allowed_transitions
import allennlp
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy

from modules import hscrf_layer_SoftDict
from metrics.span_f1 import MySpanF1
import json


@Model.register("HSCRF_SoftDict")
class HSCRF_SoftDict(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 softdict_text_field_embedder: TextFieldEmbedder,
                 softdict_encoder: Seq2SeqEncoder,
                 #softdict_feedforward: FeedForward,
                 softdict_pretrained_path: str,
                 encoder: Seq2SeqEncoder,
                 feature_size: int,
                 max_span_width: int,
                 span_label_namespace: str = "span_tags",
                 token_label_namespace: str = "token_tags",
                 feedforward: Optional[FeedForward] = None,
                 token_label_encoding: Optional[str] = None,
                 constraint_type: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 enhance: Optional[float] = None,
                 verbose_metrics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        
        super().__init__(vocab, regularizer)
        self.span_label_namespace = span_label_namespace
        self.token_label_namespace = token_label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_span_tags = self.vocab.get_vocab_size(span_label_namespace)
        self.num_token_tags = self.vocab.get_vocab_size(token_label_namespace)
        self.max_span_width = max_span_width
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        
        self.end_token_embedding = torch.nn.Parameter(torch.zeros(text_field_embedder.get_output_dim()))
        
        bias = np.sqrt( 3.0 / text_field_embedder.get_output_dim())
        torch.nn.init.uniform(self.end_token_embedding, -bias, bias)
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        if enhance:
            self.enhance = enhance
        else:
            self.enhance = None

        self._feedforward = feedforward
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
            
        softdict_length_embedder = torch.nn.Embedding(max_span_width, feature_size)
        
        #attention_W = torch.nn.Parameter(torch.zeros(size=(2 * softdict_encoder.get_output_dim(), 1)))
        softdict_linear = self.linear = torch.nn.Linear(in_features=softdict_encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('span_tags'))
        softdict_encoder = TimeDistributed(softdict_encoder)
        
        '''
        attention_W, back_dic, pos_set = self.load_weights(softdict_text_field_embedder, 
                            softdict_length_embedder,
                            softdict_encoder,
                            softdict_linear,
                            #softdict_BILOU_tag_projection_layer,
                            softdict_pretrained_path)
        '''


        self.hscrf_layer = hscrf_layer_SoftDict.HSCRF(
            ix_to_tag=copy.copy(self.vocab.get_index_to_token_vocabulary(span_label_namespace)),
            word_rep_dim=output_dim,
            ALLOWED_SPANLEN=self.max_span_width,
            softdict_text_field_embedder=softdict_text_field_embedder,
            length_embedder=softdict_length_embedder,
            encoder=softdict_encoder,
            #attention_W = attention_W,
            softdict_linear=softdict_linear,
            enhance=self.enhance,
            #back_dic=back_dic,
            #pos_set=pos_set
            #BILOU_tag_projection_layer=softdict_BILOU_tag_projection_layer
        )
        
        if constraint_type is not None:
            token_label_encoding = constraint_type
        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = token_label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = token_label_encoding is not None

        self.token_label_encoding = token_label_encoding
        if constrain_crf_decoding:
            token_labels = self.vocab.get_index_to_token_vocabulary(token_label_namespace)
            constraints = allowed_transitions(token_label_encoding, token_labels)
        else:
            constraints = None
            
        self.metrics = {}
        self.calculate_span_f1 = calculate_span_f1
        self._span_f1_metric = MySpanF1()
        #print(text_field_embedder.get_output_dim())
        #print(encoder.get_input_dim())
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)
        
        
        
    def load_weights(self, 
                     #softdict_text_field_embedder,
                     softdict_embedding,
                     softdict_length_embedder,
                     softdict_encoder,
                     softdict_linear,
                     #softdict_BILOU_tag_projection_layer,
                     pretrained_path):
        pretrained_model_state = torch.load(pretrained_path)
        #print(pretrained_model_state)
        #test = {"tokens":torch.LongTensor([[    34,     68,    600,  30016,   1583,   1098,  16550,    523,     512,   5925,    685,   4471,   8818,    128],       [    34,     68,    600,  30016,   1583,   1098,  16550,    523,     512,   5925,    685,   4471,   8818,    128]])}
        #embed = torch.from_numpy(np.load("embed.npy"))

        attention_W = pretrained_model_state["W"]
        attention_W.requires_grad = False

        #print(pretrained_model_state["word_embeddings.token_embedder_tokens.weight"]),exit(0)
        softdict_embedding_statedict = {"token_embedder_tokens.weight": pretrained_model_state["word_embeddings.token_embedder_tokens.weight"]}
        #tmp = pretrained_model_state["word_embeddings.token_embedder_tokens.weight"]
        #softdict_embedding_statedict["token_embedder_tokens.weight"] = torch.cat([tmp, tmp[0:1].expand(self.vocab.get_vocab_size('tokens') - tmp.size(0), -1)], dim=0)

        softdict_embedding.load_state_dict(softdict_embedding_statedict)
        softdict_embedding.eval()
        for param in softdict_embedding.parameters():
            '''
            with open("embed.npy", "rb") as nf:
                p1 = np.load(nf)
            print(param.equal(torch.from_numpy(p1).cpu())),exit(0)
            print(param)
            '''
            param.requires_grad = False

        
        softdict_encoder_statedict = {"_module." + ky[len("encoder")+1:]:val for ky, val in pretrained_model_state.items() if ky.startswith("encoder")}
        softdict_encoder.load_state_dict(softdict_encoder_statedict)
        softdict_encoder.eval()

        for param in softdict_encoder.parameters():
            #print(param)
            param.requires_grad = False

        softdict_linear_dict = {ky[len("linear")+1:]: val for ky, val in pretrained_model_state.items() if ky.startswith("linear")}
        softdict_linear.load_state_dict(softdict_linear_dict)
        softdict_linear.eval()

        for param in softdict_linear.parameters():
            param.requires_grad = False

        with open("./data/dict.json", "r", encoding="utf-8") as df:
            dic = json.load(df)
        
        with open("./data/pos_set.txt", "r", encoding="utf-8") as sf:
            pos_set = set([line.strip() for line in sf.readlines()])

        return attention_W, dic, pos_set

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],        # ori word sequence
                spans: torch.LongTensor,                    # double circulation to build all possible spans: (0,0),(0,1)...(0,11)...(10,11),(11,11)
                gold_spans: torch.LongTensor,               # real span sequence corresponding to word and tag sequence
                tags: torch.LongTensor = None,              # golden BIOUL tags
                span_labels: torch.LongTensor = None,       # the corresponding label of each span in line2
                gold_span_labels: torch.LongTensor = None,  # the corresponding label of each gold span in line3
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        '''
            tags: Shape(batch_size, seq_len)
                bilou scheme tags for crf modelling
        '''
        
        batch_size = spans.size(0)
        # Adding mask
        mask = util.get_text_field_mask(tokens)

        token_mask = torch.cat([mask, 
                                mask.new_zeros(batch_size, 1)],
                                dim=1)

        embedded_text_input = self.text_field_embedder(tokens)

        embedded_text_input = torch.cat([embedded_text_input, 
                                         embedded_text_input.new_zeros(batch_size, 1, embedded_text_input.size(2))],
                                        dim=1)

        # span_mask Shape: (batch_size, num_spans), 1 or 0
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        gold_span_mask = (gold_spans[:,:,0] >=0).squeeze(-1).float()
        last_span_indices = gold_span_mask.sum(-1,keepdim=True).long()

        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        batch_indices = util.move_to_device(batch_indices, 
                                            util.get_device_of(embedded_text_input))
        last_span_indices = torch.cat([batch_indices, last_span_indices],dim=-1)
        if util.get_device_of(spans) >= 0:
            embedded_text_input[last_span_indices[:,0], last_span_indices[:,1]] += self.end_token_embedding.cuda(util.get_device_of(spans))
        else:
            embedded_text_input[last_span_indices[:,0], last_span_indices[:,1]] += self.end_token_embedding

        token_mask[last_span_indices[:,0], last_span_indices[:,1]] += 1.
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.

        # spans Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()
        gold_spans = F.relu(gold_spans.float()).long()
        num_spans = spans.size(1)
        num_gold_spans = gold_spans.size(1)

        # Shape (batch_size, num_gold_spans, 4)
        hscrf_target = torch.cat([gold_spans, gold_spans.new_zeros(*gold_spans.size())],
                                 dim=-1)
        hscrf_target[:,:,2] = torch.cat([
            (gold_span_labels.new_zeros(batch_size, 1)+self.hscrf_layer.start_id).long(), # start tags in the front
            gold_span_labels.squeeze()[:,0:-1]],
            dim=-1)
        hscrf_target[:,:,3] = gold_span_labels.squeeze()
        # Shape (batch_size, num_gold_spans+1, 4)  including an <end> singular-span
        hscrf_target = torch.cat([hscrf_target, gold_spans.new_zeros(batch_size, 1, 4)],
                                 dim=1)

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],0:2] = \
                hscrf_target[last_span_indices[:,0], last_span_indices[:,1]-1][:,1:2] + 1

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],2] = \
                hscrf_target[last_span_indices[:,0], last_span_indices[:,1]-1][:,3]

        hscrf_target[last_span_indices[:,0], last_span_indices[:,1],3] = \
                self.hscrf_layer.stop_id
        
        

        # span_mask Shape: (batch_size, num_spans), 1 or 0
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()

        gold_span_mask = torch.cat([gold_span_mask.float(), 
                                gold_span_mask.new_zeros(batch_size, 1).float()], dim=-1)
        gold_span_mask[last_span_indices[:,0], last_span_indices[:,1]] = 1.


        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.

        # spans Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()
        num_spans = spans.size(1)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, token_mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)
            pass

        hscrf_neg_log_likelihood = self.hscrf_layer(
            encoded_text, 
            tokens,
            token_mask.sum(-1).squeeze(),
            hscrf_target,
            gold_span_mask
        )

        pred_results = self.hscrf_layer.get_scrf_decode(
            token_mask.sum(-1).squeeze()
        )
        #print(pred_results), print(metadata), exit(0)
        self._span_f1_metric(
            pred_results, 
            [dic['gold_spans'] for dic in metadata],
            sentences=[x["words"] for x in metadata])
        output = {
            "mask": token_mask,
            "loss": hscrf_neg_log_likelihood,
            "results": pred_results
        }
        #print(pred_results),exit(0)
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
    
    
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        if "tags" in output_dict:
            output_dict["tags"] = [
                    [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                    for tag in instance_tags]
                    for instance_tags in output_dict["tags"]
            ]
        else:
            s_e = [[(start, end) for (start, end), value in result.items() if value == 'CON'] for result in output_dict["results"]]
            word = output_dict["words"]
            output_dict = {
                "Conference_Entity": [[words[s:e+1] for (s, e) in s_and_e]for s_and_e, words in zip(s_e, word)]
            }

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        if self.calculate_span_f1:
            span_f1_dict = self._span_f1_metric.get_metric(reset=reset)
            span_kys = list(span_f1_dict.keys())
            for ky in span_kys:
                span_f1_dict[ky] = span_f1_dict.pop(ky)
            if self._verbose_metrics:
                metrics_to_return.update(span_f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in span_f1_dict.items() if
                        "overall" in x})
        return metrics_to_return
