from __future__ import print_function, division
import torch
torch.set_printoptions(threshold = 1e6)
import torch.nn as nn
from torch.autograd import Variable
import allennlp.nn.util as util
import numpy as np
import torch.nn.functional as F
from allennlp.nn.util import get_final_encoder_states
import numpy as np



class HSCRF(nn.Module):
    def __init__(self, ix_to_tag, word_rep_dim=300, SCRF_feature_dim=100, index_embeds_dim=10, ALLOWED_SPANLEN=7, 
                 softdict_text_field_embedder=None,
                 length_embedder=None,
                 encoder=None,
                 attention_W = None,
                 softdict_linear = None,
                 enhance = None,
                 back_dic=None,
                 pos_set=None
                 #BILOU_tag_projection_layer=None
                 ):
        super(HSCRF, self).__init__()
        self.ix_to_tag = ix_to_tag
        self.entity_tag_ids = [ky for ky,val in ix_to_tag.items() if val != "O"]
        self.tag_to_ix = {v:k for k,v in self.ix_to_tag.items()}
        self.tagset_size = len(ix_to_tag) + 2 # including <start, end>
        self.index_embeds_dim = index_embeds_dim
        self.SCRF_feature_dim = SCRF_feature_dim
        self.ALLOWED_SPANLEN = ALLOWED_SPANLEN
        
        self.softdict_text_field_embedder = softdict_text_field_embedder

        self.length_embedder = length_embedder
        self.encoder = encoder
        #self.BILOU_tag_projection_layer = BILOU_tag_projection_layer
        
        self.start_id = self.tagset_size - 1
        self.stop_id = self.tagset_size - 2
        
        self.tanher = nn.Tanh()
        
        self.ix_to_tag[self.start_id] = 'START'
        self.ix_to_tag[self.stop_id] = 'STOP'
        
        self.grconv = False

        self.index_embeds = nn.Embedding(self.ALLOWED_SPANLEN, self.index_embeds_dim)
        self.init_embedding(self.index_embeds.weight)

        self.dense = nn.Linear(word_rep_dim, self.SCRF_feature_dim)
        self.init_linear(self.dense)

        # 4 for SBIE, 3 for START, STOP, O and 2 for START and O
        self.CRF_tagset_size = 4*(self.tagset_size-3)+2

        self.transition = nn.Parameter(
            torch.zeros(self.tagset_size, self.tagset_size))

        self.att_dim = 0
        span_word_embedding_dim = 2*self.SCRF_feature_dim + self.index_embeds_dim + self.att_dim
        self.new_hidden2CRFtag = nn.Linear(span_word_embedding_dim, self.CRF_tagset_size)
        self.init_linear(self.new_hidden2CRFtag)
        self.LeakyReLU = torch.nn.LeakyReLU(0.1)
        self.attention_W = attention_W
        self.softdict_linear = softdict_linear
        self.enhance = enhance

        self.back_dic = back_dic
        self.pos_set = pos_set



    def init_embedding(self, input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def get_logloss_denominator(self, scores, mask):
        """
        calculate all path scores of SCRF with dynamic programming
        args:
            scores (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask   (batch_size) : mask for words
        """

        if util.get_device_of(mask) >= 0:
            logalpha = Variable(torch.FloatTensor(self.batch_size, self.sent_len+1, self.tagset_size).fill_(-10000.)).cuda(util.get_device_of(mask))
        else:
            logalpha = Variable(torch.FloatTensor(self.batch_size, self.sent_len+1, self.tagset_size).fill_(-10000.))
        logalpha[:, 0, self.start_id] = 0.
        istarts = [0] * self.ALLOWED_SPANLEN + list(range(self.sent_len - self.ALLOWED_SPANLEN+1))
        for i in range(1, self.sent_len+1):
            tmp = scores[:, istarts[i]:i, i-1] + \
                    logalpha[:, istarts[i]:i].unsqueeze(3).expand(self.batch_size, i - istarts[i], self.tagset_size, self.tagset_size)
            tmp = tmp.transpose(1, 3).contiguous().view(self.batch_size, self.tagset_size, (i-istarts[i])*self.tagset_size)
            max_tmp, _ = torch.max(tmp, dim=2)
            tmp = tmp - max_tmp.view(self.batch_size, self.tagset_size, 1)
            logalpha[:, i] = max_tmp + torch.log(torch.sum(torch.exp(tmp), dim=2))

        mask = mask.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.tagset_size)
        alpha = torch.gather(logalpha, 1, mask).squeeze(1)
        return alpha[:,self.stop_id].sum()

    def decode(self, factexprscalars, mask):
        """
        decode SCRF labels with dynamic programming
        args:
            factexprscalars (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask            (batch_size) : mask for words
        """

        batch_size = factexprscalars.size(0)
        sentlen = factexprscalars.size(1)
        factexprscalars = factexprscalars.data
        if util.get_device_of(mask) >= 0:
            logalpha = torch.FloatTensor(batch_size, sentlen+1, self.tagset_size).fill_(-10000.).cuda(util.get_device_of(mask))
            logalpha[:, 0, self.start_id] = 0.
            starts = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda(util.get_device_of(mask))
            ys = torch.zeros((batch_size, sentlen, self.tagset_size)).cuda(util.get_device_of(mask))
        else:
            logalpha = torch.FloatTensor(batch_size, sentlen+1, self.tagset_size).fill_(-10000.)
            logalpha[:, 0, self.start_id] = 0.
            starts = torch.zeros((batch_size, sentlen, self.tagset_size))
            ys = torch.zeros((batch_size, sentlen, self.tagset_size))

        for j in range(1, sentlen + 1):
            istart = 0
            if j > self.ALLOWED_SPANLEN:
                istart = max(0, j - self.ALLOWED_SPANLEN)
            f = factexprscalars[:, istart:j, j - 1].permute(0, 3, 1, 2).contiguous().view(batch_size, self.tagset_size, -1) + \
                logalpha[:, istart:j].contiguous().view(batch_size, 1, -1).expand(batch_size, self.tagset_size, (j - istart) * self.tagset_size)
            logalpha[:, j, :], argm = torch.max(f, dim=2)
            starts[:, j-1, :] = (argm / self.tagset_size + istart)
            ys[:, j-1, :] = (argm % self.tagset_size)

        batch_scores = []
        batch_spans = []
        for i in range(batch_size):
            spans = {}
            batch_scores.append(max(logalpha[i, mask[i]-1]))
            end = mask[i]-1
            y = self.stop_id
            while end >= 0:
                start = int(starts[i, end, y])
                y_1 = int(ys[i, end, y])
                if self.ix_to_tag[int(y)] not in ('START', 'STOP'):
                    spans[(int(start),int(end))] = self.ix_to_tag[int(y)]
                # spans.append((start, end, y_1, y))
                y = y_1
                end = start - 1
            batch_spans.append(spans)
            pass
        return batch_spans, batch_scores

    def get_logloss_numerator(self, goldfactors, scores, mask):
        """
        get scores of best path
        args:
            goldfactors (batch_size, tag_len, 4) : path labels
            scores      (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : all tag scores
            mask        (batch_size, tag_len) : mask for goldfactors
        """
        batch_size = scores.size(0)
        sent_len = scores.size(1)
        tagset_size = scores.size(3)
        #print("goldfactor:", goldfactors)
        # goldfactors: 25 * 39
        # factorexprs: 25 * sent_len*tagset_size
        # val: 25 * 39
        # numerator: 

        # the transform of goldfactor is for the new view of factorexprs
        goldfactors = goldfactors[:, :, 0]*sent_len*tagset_size*tagset_size + goldfactors[:,:,1]*tagset_size*tagset_size+goldfactors[:,:,2]*tagset_size+goldfactors[:,:,3]
        factorexprs = scores.view(batch_size, -1)
        # select the path recorded in goldfactors
        #print("--------------------------\n",goldfactors)
        #print("--------------------------\n",factorexprs[:,:10000])
        #print(factorexprs.size(), goldfactors.size()),exit(0)
        val = torch.gather(factorexprs, 1, goldfactors)
        # choose span-level unmasked
        numerator = val.masked_select(mask.byte())
        #print("goldfactor:", goldfactors, goldfactors.size()), print("factorexprs:", factorexprs, factorexprs.size()), print("val:", val, val.size()), print("numerator:", numerator, numerator.size()), exit(0)
        return numerator


    def HSCRF_scores(self, global_feats, token_indices):
        ### TODO: need to improve
        """
        calculate SCRF scores with HSCRF
        args:
            global_feats (batch_size, sentence_len, featsdim) : word representations
        """

        # 3 for O, STOP, START
        validtag_size = self.tagset_size-3
        if util.get_device_of(global_feats) >= 0:
            scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size)).cuda(util.get_device_of(global_feats))
            diag0 = torch.LongTensor(range(self.sent_len)).cuda(util.get_device_of(global_feats))
            # m10000 for STOP
            m10000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 1)).cuda(util.get_device_of(global_feats))
            # m30000 for O, START, STOP
            m30000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 3)).cuda(util.get_device_of(global_feats))
            #print(m10000.size()), print(m30000.size()), print(diag0.size(), diag0)
        else:
            scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size))
            diag0 = torch.LongTensor(range(self.sent_len))
            # m10000 for STOP
            m10000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 1))
            # m30000 for O, START, STOP
            m30000 = Variable(torch.FloatTensor([-10000.]).expand(self.batch_size, self.sent_len, self.tagset_size, 3))
            
        for span_len in range(min(self.ALLOWED_SPANLEN, self.sent_len-1)):
            #if span_len != 34: continue

            #emb_x, att_logits = self.concat_features(global_feats, token_indices, span_len)
            emb_x = self.concat_features(global_feats, token_indices, span_len)
            
            #att_logits = att_logits.unsqueeze(-1)
            # CRF2tag is the weight parameter vector for token label
            emb_x = self.new_hidden2CRFtag(emb_x)
            #print(emb_x.size())
            #emb_x [batch_size, span_len, span_seq_len, tag_size], [8,1,224,6]
            #print(self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).size())
            #print(att_logits.size())
            #emb_x[:, 0, :, :validtag_size] [8, 224, 1]
            
            if span_len == 0:
                tmp = torch.cat(
                                    (   # choose 4 valid tags and unsqueeze the transition to batch_size * sent_len * (tag * tag) by 2 unsqueeze(0)
                                        self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0) + \
                                        # [1,1,tag_size,valid_tag_size], [1,1,4,1]
                                        emb_x[:, 0, :, :validtag_size].unsqueeze(2),
                                        # [batch_size, 0, span_seq_len, valid_tag_size], [8,224,1,1]
                                        self.transition[:, -2:].unsqueeze(0).unsqueeze(0) + emb_x[:, 0, :, -2:].unsqueeze(2),
                                        m10000
                                    ),
                                    3
                                )
                # sent_len * sent_len each points is a tag transition
                scores[:, diag0, diag0] = tmp
            elif span_len == 1:
                tmp = torch.cat(
                                    (   #expand to new size, expand can only be applied on the size=1 dimension
                                        self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-1, self.tagset_size, validtag_size) + \
                                        (emb_x[:, 0, :, validtag_size:2*validtag_size] + emb_x[:, 1, :, 3*validtag_size:4*validtag_size]).unsqueeze(2),
                                        m30000[:, 1:]
                                    ), 
                                    3
                                )
                scores[:, diag0[:-1], diag0[1:]] = tmp

            elif span_len == 2:
                tmp = torch.cat(
                                    (
                                        self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-2, self.tagset_size, validtag_size) + \
                                        (emb_x[:, 0, :, validtag_size:2*validtag_size] + emb_x[:, 1, :, 2*validtag_size:3*validtag_size] + emb_x[:, 2, :, 3*validtag_size:4*validtag_size]).unsqueeze(2),
                                        m30000[:, 2:]
                                    ),
                                    3
                                )
                scores[:, diag0[:-2], diag0[2:]] = tmp

            elif span_len >= 3:
                tmp0 = self.transition[:, :validtag_size].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.sent_len-span_len, self.tagset_size, validtag_size) + \
                        (emb_x[:, 0, :, validtag_size:2*validtag_size] + emb_x[:, 1:span_len, :, 2*validtag_size:3*validtag_size].sum(1) + emb_x[:, span_len,:, 3*validtag_size:4*validtag_size]).unsqueeze(2)
                        # emb_x dim of -1 equals to CRF tag number not span tag number, validtag_size, 2, 3, 4*validtag_size is BIES token tag
                tmp = torch.cat(
                                    (
                                        tmp0,
                                        m30000[:, span_len:]
                                    ),
                                    3
                                )
                scores[:, diag0[:-span_len], diag0[span_len:]] = tmp
            
            del tmp
            if "tmp0" in locals().keys():
                del tmp0
            #print(span_len), print(tmp.size(), tmp)

            '''
            if span_len < 15:
                continue
            add_dim = span_len
            #print(att_logits)
            #att_logits = torch.ge(att_logits, torch.full_like(att_logits, 0.7)).float()
            att_logits = torch.where(att_logits < 0.7, torch.zeros_like(att_logits), torch.full_like(att_logits, 100.0))
            #print(att_logits)
            if util.get_device_of(att_logits) >= 0:
                addpart = torch.zeros(self.batch_size, add_dim).cuda(util.get_device_of(att_logits))
            else:
                addpart = torch.zeros(self.batch_size, add_dim)
            att_logits = torch.cat([addpart, att_logits, addpart], -1)
            #print(att_logits.size())

            if span_len == 0:
                att_logits = torch.Tensor([np.diag(each) for each in att_logits.cpu()]).unsqueeze(-1).unsqueeze(-1).expand([self.batch_size, self.sent_len, self.sent_len, validtag_size, validtag_size])
            else:
                att_logits = torch.Tensor([np.diag(each) for each in att_logits.cpu()])[:, add_dim:, :-add_dim].unsqueeze(-1).unsqueeze(-1).expand([self.batch_size, self.sent_len, self.sent_len, validtag_size, validtag_size])
            att_logits = torch.cat([torch.zeros(self.batch_size, self.sent_len, self.sent_len, validtag_size, validtag_size), att_logits, torch.zeros(self.batch_size, self.sent_len, self.sent_len, 2, validtag_size)], -2)
            
            if util.get_device_of(scores) >= 0:
                att_logits = torch.cat([att_logits, torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size-validtag_size)], -1).cuda(util.get_device_of(scores))
            else:
                att_logits = torch.cat([att_logits, torch.zeros(self.batch_size, self.sent_len, self.sent_len, self.tagset_size, self.tagset_size-validtag_size)], -1)
            #print(att_logits, att_logits.size())
            scores = scores + att_logits
            del att_logits
            '''
            
            

        # scores      (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : all tag scores
        # transition on different span lengths (from pos i to pos j) and different tag
        #exit(0)
        return scores

    def concat_features(self, emb_z, token_indices, span_len):
        """
        concatenate two features
        args:
            emb_z (batch_size, sentence_len, featsdim) : contextualized word representations
            token_indices: Dict[str, LongTensor], indices of different fields
            span_len: a number (from 0)
        """
        #print(token_indices),exit(0)
        batch_size = emb_z.size(0)
        sent_len = emb_z.size(1)
        hidden_dim = emb_z.size(2)
        emb_z = emb_z.unsqueeze(1).expand(batch_size, 1, sent_len, hidden_dim)
        
        span_exprs = [emb_z[:, :, i:i + span_len + 1] for i in range(sent_len - span_len)]
        span_exprs = torch.cat(span_exprs, 1)
        
        endpoint_vec = (span_exprs[:, :, 0]-span_exprs[:, :, span_len]).unsqueeze(2).expand(batch_size, sent_len-span_len, span_len+1, hidden_dim)
        
        if util.get_device_of(emb_z) >= 0:
            index = Variable(torch.LongTensor(range(span_len+1))).cuda(util.get_device_of(emb_z))
        else:
            index = Variable(torch.LongTensor(range(span_len+1)))
        index = self.index_embeds(index).unsqueeze(0).unsqueeze(0).expand(batch_size, sent_len-span_len, span_len+1, self.index_embeds_dim)
        
        # TODO: calculate BILOU features
        #BILOU_features = self.get_BILOU_features(token_indices, sent_len, span_len)
        #BILOU_features = span_exprs.new_zeros(batch_size, span_exprs.size(1),span_exprs.size(2),16)

        #if span_len >= 10:
        #attention_feature, attention_logits = self.get_BILOU_features(token_indices, sent_len, span_len)
        
        #print(attention_feature)
        #else:
        #attention_feature = span_exprs.new_zeros(batch_size, span_exprs.size(1), span_exprs.size(2), self.att_dim)
        
        #print(attention_feature)
        
        # In HSCRFs, v'_i is the concatenation of (1) BiLSTM encoded representation vi, (2) v_u_j - v_t_j, and (3) emb(i - t_j + 1), the position embedding in the segment.
        # soft-dict layer: U_i = cat[N_i, v_i]
        #new_emb = torch.cat((span_exprs, BILOU_features, endpoint_vec, index), 3)
        #new_emb = torch.cat((span_exprs, attention_feature, endpoint_vec, index), 3)
        new_emb = torch.cat((span_exprs, endpoint_vec, index), 3)
        #new_emb = new_emb * attention_feature.expand_as(new_emb)
        #return new_emb.transpose(1,2).contiguous(), attention_logits
        return new_emb.transpose(1,2).contiguous()
    

    def attention(self, lstm_output, final_state, mask_cuda=0):
        
        new_final_state = torch.unsqueeze(final_state,1).expand(final_state.size()[0], lstm_output.size()[1], final_state.size()[-1])
        merge = torch.cat([lstm_output, new_final_state], -1)
        #print(merge)
        #print(self.attention_W)
        e = self.LeakyReLU(torch.matmul(merge.cpu(), self.attention_W.cpu())).squeeze(-1)
        # attention is the attention coefficient
        
        if mask_cuda >= 0:
            attention = F.softmax(e, dim=1).unsqueeze(1).cuda(mask_cuda)
        else:
            attention = F.softmax(e, dim=1).unsqueeze(1)
        #print(attention)
        # weighted sum by attention coefficient for each sentence
        attention_out = torch.matmul(attention, lstm_output).squeeze(1)

        logits = F.softmax(self.softdict_linear(attention_out), dim=-1)
        
        return attention.squeeze(1), attention_out, logits


    def id2words(self, token):
        words = ["".join([self.back_dic[str(word)] for word in span]) for span in token]
        return words

    
    def span_score(self, spans):
        return [int(span in self.pos_set) for span in spans]


    def get_BILOU_features(self, token_indices, sent_len, span_len):

        #print(token_indices)
        span_level_token_indices = {}        
        for ky,val in list(token_indices.items()):
            if ky == 'elmo':
                continue
            val = val.unsqueeze(1)
            span_level_token_indices[ky] = torch.cat([val[:, :, i:i + span_len + 1] for i in range(sent_len - 1 - span_len)], 1)

        '''
        print(span_level_token_indices)
        t = span_level_token_indices["tokens"][0].cpu().numpy().tolist()
        import json
        with open("./data/dict.json", "r", encoding="utf-8") as df:
            dic = json.load(df)
        a = [[dic[str(word)] for word in span] for span in t]
        print(a)
        '''
        ori_seq = [self.id2words(each.cpu().numpy().tolist()) for each in span_level_token_indices["tokens"]]
        att_logits = torch.Tensor([self.span_score(seq) for seq in ori_seq])


        spans_embedded = self.softdict_text_field_embedder(span_level_token_indices, num_wrapping_dims=1)
        spans_mask = util.get_text_field_mask(span_level_token_indices, num_wrapping_dims=1)
        
        '''
        for param in self.softdict_text_field_embedder.parameters():
            #np.save("embed.npy", param.detach().numpy())
            print(param.size()), exit(0)
        '''
        
        #print(spans_mask)
        #print(spans_mask.size())
        
        if util.get_device_of(spans_mask) >= 0:
            att_mask = torch.ge(torch.mean(spans_mask.float(), -1), (torch.ones(spans_mask.size(0), spans_mask.size(1)) - 2e-6).cuda(util.get_device_of(spans_mask)))
        else:
            att_mask = torch.ge(torch.mean(spans_mask.float(), -1), (torch.ones(spans_mask.size(0), spans_mask.size(1)) - 2e-6))
                
        dim_2_pad = self.ALLOWED_SPANLEN - spans_embedded.size(2)
        p2d = (0,0,0, dim_2_pad)
        # now shape (batch_size, num_span, max_span_width, dim)
        spans_embedded = F.pad(spans_embedded, p2d, "constant", 0.)
        spans_mask = F.pad(spans_mask, (0, dim_2_pad), "constant", 0.)
        #print("embed:")
        #print(spans_embedded)

        '''
        tt = {"tokens":torch.LongTensor([   50,  1138,    84,     7,   645,  1135,  7386,  1123,  4979,   952,
             2,   381,   173,   128,  8932,     9,    95,  1098, 16550,   524,
          3897,  5190,  8242,    22,  2112,  6912,  1408,   814,  9853,   128])}
        t = self.softdict_text_field_embedder(tt)
        print(t),exit(0)
        '''
        
        
        batch_size = spans_mask.size(0)
        num_spans = spans_mask.size(1)
        if util.get_device_of(spans_mask) >= 0:
            length_vec = torch.autograd.Variable(torch.LongTensor(range(self.ALLOWED_SPANLEN))).cuda(util.get_device_of(spans_mask))
        else:
            length_vec = torch.autograd.Variable(torch.LongTensor(range(self.ALLOWED_SPANLEN)))
        length_vec = self.length_embedder(length_vec).unsqueeze(0).unsqueeze(0).expand(batch_size, num_spans, -1,-1)
        
        spans_encoded = self.encoder(spans_embedded, spans_mask)        #BiLSTM
        
        
        #spans_encoded = torch.cat((spans_encoded, length_vec), 3).contiguous()
        #print(spans_encoded)

        spans_encoded = spans_encoded.reshape([batch_size * num_spans, self.ALLOWED_SPANLEN, -1])
        '''
            [batch_size * num_spans, self.ALLOWED_SPANLEN] shaped mask may occur whole zero
            like tensor([[1, 0, 0,  ..., 0, 0, 0],
                    [1, 0, 0,  ..., 0, 0, 0],
                    [1, 0, 0,  ..., 0, 0, 0],
                    ...,
                    [0, 0, 0,  ..., 0, 0, 0],
                    [0, 0, 0,  ..., 0, 0, 0],
                    [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0'), and use 'get_final_encoder_states()' will lead to
                    the error RuntimeError: cuda runtime error (59) : device-side assert triggered at /pytorch/aten/src/THC/THCReduceAll.cuh:327

            change to
            the tensor masked on span_sequence level is still assign as the unmasked tensor: [1 * 1:0*other] remain one 1
        
        '''
        spans_mask = spans_mask.reshape([batch_size * num_spans, self.ALLOWED_SPANLEN])
        if util.get_device_of(spans_mask) >= 0:
            tmp = torch.zeros(self.ALLOWED_SPANLEN, dtype=torch.int64).cuda(util.get_device_of(spans_mask))
        else:
            tmp = torch.zeros(self.ALLOWED_SPANLEN, dtype=torch.int64)
        tmp[0] = 1
        tmp = tmp.expand([batch_size * num_spans, self.ALLOWED_SPANLEN])
        
        new_spans_mask = spans_mask | tmp
        #print(new_spans_mask)
        last_state = get_final_encoder_states(spans_encoded, new_spans_mask)
        attention_coe, attention_out, attention_logits = self.attention(lstm_output=spans_encoded, final_state=last_state, mask_cuda=util.get_device_of(spans_mask))
        #print(attention_logits),exit(0)
        attention_logits = attention_logits.reshape([batch_size, num_spans, -1])[:,:,1]     # here 0 stand for true / 1
        #print(attention_logits)
        #print(attention_logits.size())
        attention_logits = attention_logits * att_mask.float()
        #print(attention_logits), exit(0)
        attention_out = attention_out.reshape([batch_size, num_spans, -1])
        
        #print(attention_coe.size())
        #attention_coe = attention_coe * spans_mask.float()
        attention_coe = attention_coe.reshape([batch_size, num_spans, -1])
        attention_coe = attention_coe.unsqueeze(-1)
        #print(attention_coe)
        #attention_coe = torch.gt(attention_coe, 0.1).float()
        attention_coe = attention_coe.expand([batch_size, num_spans, attention_coe.size(2), 1])
        #print(attention_coe.size()), exit(0)
        attention_coe = torch.cat([attention_coe, attention_coe.new_zeros(batch_size, 1, attention_coe.size(2), attention_coe.size(3))], dim=1)
        attention_out = torch.cat([attention_out, attention_out.new_zeros(batch_size, 1, attention_out.size(-1))], dim=1)
        attention_logits = torch.cat([attention_logits, attention_logits.new_zeros(batch_size, 1)], dim=1)

        #print(attention_logits.size(), att_logits.size()),exit(0)
        att_logits = torch.cat([att_logits, att_logits.new_zeros(batch_size, 1)], dim=1)
        return attention_coe[:,:,:span_len+1,:].detach(), att_logits
        #return attention_out.unsqueeze(2).expand([batch_size, num_spans+1, span_len+1, attention_out.size(-1)]).detach(), attention_logits.unsqueeze(-1)
        

    def forward(self, feats, token_indices, mask_word, tags, mask_tag):
        """
        calculate loss
        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask_word (batch_size) : sentence lengths
            tags (batch_size, tag_len, 4) : target
            mask_tag (batch_size, tag_len) : tag_len <= sentence_len
        """
        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        
        self.SCRF_scores = self.HSCRF_scores(feats, token_indices)
        #print(self.SCRF_scores)
        forward_score = self.get_logloss_denominator(self.SCRF_scores, mask_word)
        numerator = self.get_logloss_numerator(tags, self.SCRF_scores, mask_tag)
        
        return (forward_score - numerator.sum()) / self.batch_size  # loss = -log(n/d) = log d - log n

    def get_scrf_decode(self, mask):
        """
        decode with SCRF
        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask  (batch_size) : mask for words
        """
        #print(self.SCRF_scores)
        batch_spans, batch_scores = self.decode(self.SCRF_scores, mask)
        return batch_spans
    
    
    def get_npy(self):                        
        for param in self.softdict_text_field_embedder.parameters():
            with open("embed.npy", "rb") as nf:
                p1 = np.load(nf)
            print(param.equal(torch.from_numpy(p1).cpu()))
    