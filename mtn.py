import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
from data_utils import * 

class EncoderDecoder(nn.Module):
    def __init__(self, mode, query_encoder, his_encoder, cap_encoder, vid_encoder, decoder, query_embed, his_embed, cap_embed, tgt_embed, generator, diff_encoder=False, auto_encoder_embed=None, auto_encoder_ft=None, auto_encoder_generator=None):
        super(EncoderDecoder, self).__init__()
        self.mode = mode 
        self.query_encoder = query_encoder
        self.his_encoder = his_encoder
        self.cap_encoder = cap_encoder
        self.vid_encoder = vid_encoder
        self.decoder = decoder
        self.query_embed = query_embed
        self.his_embed = his_embed
        self.cap_embed = cap_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.diff_encoder = diff_encoder
        self.auto_encoder_embed = auto_encoder_embed
        self.auto_encoder_ft=auto_encoder_ft
        self.auto_encoder_generator=auto_encoder_generator

    def forward(self, b):
        video_features, video_features_mask, cap, his, his_st, query, tgt, cap_mask, his_mask, query_mask, tgt_mask = b.fts, b.fts_mask, b.cap, b.his, b.his_st, b.query, b.trg, b.cap_mask, b.his_mask, b.query_mask, b.trg_mask
        auto_encoded_ft = None
        if self.mode in [1,2]: 
            encoded_his = self.his_encode(his, his_st, his_mask)
        else: 
            encoded_his = None 
        
        if self.mode in [2]:
            encoded_cap = self.cap_encode(cap, cap_mask)
        else:
            encoded_cap = None
        
        if self.mode in [3]:
            encoded_query, encoded_cap, encoded_his = self.query_encode(query, query_mask, his, his_mask, cap, cap_mask)
        elif self.mode in [4]:
            encoded_query, encoded_vid_features, encoded_cap, encoded_his, auto_encoded_ft = self.query_encode(query, query_mask, his, his_mask, cap, cap_mask, video_features, video_features_mask)
        else:
            encoded_query = self.query_encode(query, query_mask)

        if self.mode not in [4]:
            encoded_vid_features = None 

        cap2res_mask = None
        return self.decode(encoded_vid_features, encoded_his, encoded_cap, encoded_query, video_features_mask, his_mask, cap_mask, query_mask, tgt, tgt_mask, cap2res_mask, auto_encoded_ft)

    def vid_encode(self, video_features, video_features_mask, encoded_query=None):
        output = []
        for i, ft in enumerate(video_features):
            output.append(self.vid_encoder[i](ft))
        return output

    def query_encode(self, query, query_mask, encoded_his=None, his_mask=None, encoded_cap=None, cap_mask=None, vid=None, vid_mask=None):
        if self.mode in [3]:
            cap = encoded_cap
            his = encoded_his
            return self.query_encoder(self.query_embed(query), self.query_embed(cap), self.query_embed(his))
        elif self.mode in [4]:
            cap = encoded_cap
            his = encoded_his
            if self.diff_encoder:
                if self.auto_encoder_ft == 'caption' or self.auto_encoder_ft == 'summary':
                    ft = cap
                elif self.auto_encoder_ft == 'query':
                    ft = query
                if self.auto_encoder_embed is not None:
                    ae_encoded = []
                    for i in range(len(vid)):
                        ae_encoded.append(self.auto_encoder_embed[i](ft))
                else:
                    ae_encoded = []
                    for i in range(len(vid)):
                        ae_encoded.append(self.query_embed(ft))
                return self.query_encoder(self.query_embed(query), self.vid_encode(vid, vid_mask), self.query_embed(cap), self.query_embed(his), ae_encoded)
            else:
                output = self.query_encoder(self.query_embed(query), self.vid_encode(vid, vid_mask), self.query_embed(cap), self.query_embed(his))
                output.append(None)
                return output 
        else:
            return self.query_encoder(self.query_embed(query), query_mask, encoded_his, his_mask, encoded_cap, cap_mask)

    def his_encode(self, his, his_st, his_mask, seq2=None, seq2_mask=None):
        if self.his_embed is not None:
            if his_st is not None and len(his_st)>0:
                return self.his_encoder(self.his_embed(his, his_st), his_mask, seq2, seq2_mask)
            else:
                return self.his_encoder(self.his_embed(his), his_mask, seq2, seq2_mask)
        else:
            return self.his_encoder(self.query_embed(his), his_mask, seq2, seq2_mask)

    def cap_encode(self, cap, cap_mask, seq2=None, seq2_mask=None):
        if self.cap_embed is not None:
            return self.cap_encoder(self.cap_embed(cap), cap_mask, seq2, seq2_mask)
        else:
            return self.cap_encoder(self.query_embed(cap), cap_mask, seq2, seq2_mask)

    def decode(self, encoded_vid_features, his_memory, cap_memory, query_memory, vid_features_mask, his_mask, cap_mask, query_mask, tgt, tgt_mask, cap2res_mask, auto_encoded_ft):
        encoded_tgt = self.tgt_embed(tgt)
        return self.decoder(encoded_vid_features, vid_features_mask, encoded_tgt, his_memory, his_mask, cap_memory, cap_mask, query_memory, query_mask, tgt_mask, cap2res_mask, auto_encoded_ft, self.auto_encoder_ft)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, mode):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.mode = mode  
        
    def forward(self, seq1, seq1_mask, seq2=None, seq2_mask=None, seq3=None, seq3_mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            if seq2 is not None and seq3 is None: #with cross attention 
                seq1 = layer(seq1, seq1_mask, seq2, seq2_mask)
            elif seq2 is not None and seq3 is not None: #with cross attention
                seq1 = layer(seq1, seq1_mask, seq3, seq3_mask, seq2, seq2_mask)
            else: # without cross attention 
                seq1 = layer(seq1, seq1_mask)
        return self.norm(seq1)

class NoAttEncoder(nn.Module):
    def __init__(self, size, nb_layers):
        super(NoAttEncoder, self).__init__()
        self.norm = nn.ModuleList()
        self.nb_layers = nb_layers
        for n in range(nb_layers):
            self.norm.append(LayerNorm(size))

    def forward(self, *seqs):
        output = []
        i=0
        seq_i=0
        while(True):
            if isinstance(seqs[seq_i],list):
                output_seq = []
                for seq in seqs[seq_i]:
                    output_seq.append(self.norm[i](seq))
                    i+=1
                output.append(output_seq)
                seq_i+=1
            else:
                output.append(self.norm[i](seqs[seq_i]))
                i+=1
                seq_i+=1
            if i==self.nb_layers:
                break
        return output 

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out 

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ff1, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff1 = ff1
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 

    def forward(self, seq, seq_mask):
        seq = self.sublayer[0](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        return self.sublayer[1](seq, self.ff1)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, mode, ft_sizes=None):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if mode == 4:
            self.ae_norm = nn.ModuleList()
            for ft_size in ft_sizes:
                self.ae_norm.append(LayerNorm(layer.size))
        self.mode = mode

    def forward(self, vid_ft, vid_mask, x, his_memory, his_mask, cap_memory, cap_mask, query_memory, query_mask, tgt_mask, cap2res_mask, auto_encoded_ft, auto_encoded_features):
        for layer in self.layers:
            if self.mode in [1]: # attention on query as well as dialog history 
                x = layer(x, his_memory, his_mask, query_memory, query_mask, tgt_mask)
            elif self.mode in [2, 3]: # attention on query + dialog history + caption 
                x = layer(x, cap_memory, cap_mask, his_memory, his_mask, query_memory, query_mask, tgt_mask)
            elif self.mode in [4]:
                x, auto_encoded_ft = layer(x, cap_memory, cap_mask, his_memory, his_mask, query_memory, query_mask, tgt_mask, vid_ft, vid_mask, auto_encoded_ft, auto_encoded_features)
            else:  
                x = layer(x, query_memory, query_mask, tgt_mask)
        if self.mode == 4:
            out_ae_ft = []
            for i, ft in enumerate(auto_encoded_ft):
                out_ae_ft.append(self.ae_norm[i](ft))
            return self.norm(x), out_ae_ft
        else:
            return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Mode1DecoderLayer(DecoderLayer):
    def __init__(self, size, self_attn, his_attn, q_attn, feed_forward, dropout):
        super(Mode1DecoderLayer, self).__init__(size, self_attn, q_attn, feed_forward, dropout)
        self.his_attn = his_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 4)

    def forward(self, x, his_memory, his_mask, q_memory, q_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.his_attn(x, his_memory, his_memory, his_mask))
        x = self.sublayer[2](x, lambda x: self.src_attn(x, q_memory, q_memory, q_mask))
        return self.sublayer[3](x, self.feed_forward)

class Mode4DecoderLayer(DecoderLayer):
    def __init__(self, size, self_attn, cap_attn, his_attn, q_attn, feed_forward, dropout):
        super(Mode4DecoderLayer, self).__init__(size, self_attn, q_attn, feed_forward, dropout)
        self.his_attn = his_attn
        self.cap_attn = cap_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 5)

    def forward(self, x, cap_memory, cap_mask, his_memory, his_mask, q_memory, q_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cap_attn(x, cap_memory, cap_memory, cap_mask))
        x = self.sublayer[2](x, lambda x: self.his_attn(x, his_memory, his_memory, his_mask))
        x = self.sublayer[3](x, lambda x: self.src_attn(x, q_memory, q_memory, q_mask))
        return self.sublayer[4](x, self.feed_forward)

class Mode22DecoderLayer(DecoderLayer):
    def __init__(self, size, self_attn, cap_attn, his_attn, q_attn, auto_encoder_self_attn, auto_encoder_vid_attn, auto_encoder_attn, feed_forward, auto_encoder_feed_forward, dropout, auto_encoder_ff_before):
        super(Mode22DecoderLayer, self).__init__(size, self_attn, q_attn, feed_forward, dropout)
        self.his_attn = his_attn
        self.cap_attn = cap_attn
        self.auto_encoder_attn = auto_encoder_attn
        self.auto_encoder_self_attn = auto_encoder_self_attn
        self.auto_encoder_vid_attn = auto_encoder_vid_attn
        self.auto_encoder_feed_forward = auto_encoder_feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 5 + 4*len(auto_encoder_vid_attn))
        self.auto_encoder_ff_before = auto_encoder_ff_before

    def forward(self, x, cap_memory, cap_mask, his_memory, his_mask, q_memory, q_mask, tgt_mask, vid_fts, vid_mask, ae_fts, ae_features):
        count = 0
        x = self.sublayer[count](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        count += 1
        x = self.sublayer[count](x, lambda x: self.his_attn(x, his_memory, his_memory, his_mask))
        count += 1
        if ae_features == 'caption' or ae_features == 'summary':
            x = self.sublayer[count](x, lambda x: self.src_attn(x, q_memory, q_memory, q_mask))
            count += 1
            x = self.sublayer[count](x, lambda x: self.cap_attn(x, cap_memory, cap_memory, cap_mask))
            count += 1
            if ae_fts is None:
                ae_fts = cap_memory
            ae_mask = cap_mask
        elif ae_features == 'query':
            x = self.sublayer[count](x, lambda x: self.cap_attn(x, cap_memory, cap_memory, cap_mask))
            count += 1
            x = self.sublayer[count](x, lambda x: self.src_attn(x, q_memory, q_memory, q_mask))
            count += 1
            if ae_fts is None:
                ae_fts = q_memory
            ae_mask = q_mask
        out_ae_fts = []
        for i, vid_ft in enumerate(vid_fts):
            if type(ae_fts) == list:
                ae_ft = ae_fts[i]
            else:
                ae_ft = ae_fts
            ae_ft = self.sublayer[count](ae_ft, lambda ae_ft: self.auto_encoder_self_attn[i](ae_ft, ae_ft, ae_ft, ae_mask))
            count += 1
            ae_ft = self.sublayer[count](ae_ft, lambda ae_ft: self.auto_encoder_vid_attn[i](ae_ft, vid_ft, vid_ft, vid_mask[i]))
            count += 1
            if self.auto_encoder_ff_before:
                ae_ft = self.sublayer[count](ae_ft, self.auto_encoder_feed_forward[i])
                count += 1
            x = self.sublayer[count](x, lambda x: self.auto_encoder_attn[i](x, ae_ft, ae_ft, ae_mask))
            count += 1
            if not self.auto_encoder_ff_before:
                ae_ft = self.sublayer[count](ae_ft, self.auto_encoder_feed_forward[i])
                count += 1
            out_ae_fts.append(ae_ft)
        return self.sublayer[count](x, self.feed_forward), out_ae_fts


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        if d_in < 0: 
            d_in = d_model 
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model 
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class StPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=50):
        super(StPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
            
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x, x_st):
        x = x + Variable(self.pe[:, x_st], requires_grad=False)
        x = x.squeeze(0)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, 
    N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mode=-1, 
    separate_his_embed=False, separate_cap_embed=False, 
    ft_sizes=None, 
    diff_encoder=False, diff_embed=False, diff_gen=False, 
    auto_encoder_ft=None, auto_encoder_attn=False,
    auto_encoder_ff_before=False):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    generator=Generator(d_model, tgt_vocab)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    tgt_embed = [Embeddings(d_model, tgt_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = nn.Sequential(*tgt_embed)
    if separate_his_embed:
        his_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        his_embed = None 
    if separate_cap_embed:
        cap_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        cap_embed = None 
    cap_encoder = None 
    vid_encoder = None 
    his_encoder = None 
    auto_encoder_generator = None
    auto_encoder_embed = None
    if mode == 1: # query + dialogue history as source 
        his_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        query_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        decoder = Decoder(Mode1DecoderLayer(d_model, c(attn), c(attn), c(attn), c(ff), dropout), N, mode)
    elif mode == 2: # query + dialog history + caption as source, query+his+cap attention in decoder
        cap_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        his_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        query_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        decoder = Decoder(Mode4DecoderLayer(d_model, c(attn), c(attn), c(attn), c(attn), c(ff), dropout), N, mode)
    elif mode == 3: # his (no att) + cap (no att) + query (no att) as source, cap+his+query att in decoder 
        query_encoder=NoAttEncoder(d_model, nb_layers=3)
        decoder = Decoder(Mode4DecoderLayer(d_model, c(attn), c(attn), c(attn), c(attn), c(ff), dropout), N, mode)
    elif mode == 4:
        if diff_embed:
            auto_encoder_embed = nn.ModuleList()
            for ft_size in ft_sizes:
                embed = [Embeddings(d_model, src_vocab), c(position)]
                auto_encoder_embed.append(nn.Sequential(*embed))
        else:
            auto_encoder_embed = None
        if diff_encoder:
            query_encoder=NoAttEncoder(d_model, nb_layers=3 + 2*len(ft_sizes))
        else:
            query_encoder=NoAttEncoder(d_model, nb_layers=3 + len(ft_sizes))
        self_attn = nn.ModuleList()
        vid_attn = nn.ModuleList()
        ae_ff = nn.ModuleList()
        vid_encoder=nn.ModuleList()
        if auto_encoder_attn:
            auto_encoder_attn_ls = nn.ModuleList()
        else:
            auto_encoder_attn_ls = None
        for ft_size in ft_sizes:
            ff_layers = [nn.Linear(ft_size, d_model), nn.ReLU(), c(position)]
            vid_encoder.append(nn.Sequential(*ff_layers))
            self_attn.append(c(attn))
            vid_attn.append(c(attn))
            ae_ff.append(c(ff))
            if auto_encoder_attn:
                auto_encoder_attn_ls.append(c(attn))
        if diff_gen:
            auto_encoder_generator = nn.ModuleList()
            for ft_size in ft_sizes:
              auto_encoder_generator.append(c(generator))
        else:
            auto_encoder_generator = None
        decoder = Decoder(Mode22DecoderLayer(d_model, c(attn), c(attn), c(attn), c(attn), self_attn, vid_attn, auto_encoder_attn_ls, c(ff), ae_ff, dropout, auto_encoder_ff_before), N, mode, ft_sizes)
    else: # query ony as source 
        query_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, mode)
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N, mode)
    model = EncoderDecoder(
          mode = mode,
          query_encoder=query_encoder, 
          his_encoder=his_encoder,
          cap_encoder=cap_encoder,
          vid_encoder=vid_encoder,
          decoder=decoder,
          query_embed=query_embed,
          his_embed=his_embed,
          cap_embed=cap_embed,
          tgt_embed=tgt_embed,
          generator=generator,
          auto_encoder_generator=auto_encoder_generator,
          auto_encoder_embed=auto_encoder_embed,
          diff_encoder=diff_encoder,
          auto_encoder_ft=auto_encoder_ft)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
