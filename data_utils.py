import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb 
from torchtext import data, datasets 

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def seq1toseq2_mask(seq1, seq2, pad):
    temp = (seq1 != pad).unsqueeze(-1).expand((seq1.shape[0], seq1.shape[1], seq2.shape[-1]))
    output = temp & (seq2 != pad).unsqueeze(-2).expand((seq2.shape[0], seq1.shape[1], seq2.shape[-1]))
    return output

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, query, his, his_st, fts=None, cap=None, trg=None, trg_y=None, pad=0):
        self.query = query
        self.his = his
        self.his_st = his_st 
        if fts is not None:
            permuted_fts = [torch.from_numpy(ft).float().cuda().permute(1,0,2) for ft in fts]
            self.fts_mask = [(torch.sum(permuted_ft != 1, dim=2) != 0).unsqueeze(-2) for permuted_ft in permuted_fts]
            self.fts = [ ft * self.fts_mask[i].squeeze().unsqueeze(-1).expand_as(ft).float() for i, ft in enumerate(permuted_fts)]
        else:
            self.fts = None
            self.fts_mask = None 
        self.query_mask = (query != pad).unsqueeze(-2)
        self.his_mask  = (his != pad).unsqueeze(-2)
        if cap is not None:
            self.cap = cap
            self.cap_mask = (cap != pad).unsqueeze(-2)
        else:
            self.cap = None
            self.cap_mask = None
        if trg is not None:
            self.trg = trg
            self.trg_y = trg_y
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, ae_generator, criterion, opt=None, l=1.0):
        self.generator = generator
        self.ae_generator= ae_generator
        self.criterion = criterion
        self.opt = opt 
        self.l = l
    
    def __call__(self, x, y, norm, ae_x=None, ae_y=None, ae_norm=None):
        out = self.generator(x)
        loss = self.criterion(out.contiguous().view(-1, out.size(-1)), 
                              y.contiguous().view(-1)) / norm.float()
        if ae_x is not None:
            if type(ae_x) == list:
                for i, ae_in in enumerate(ae_x):
                    if self.ae_generator is not None:
                        ae_out = self.ae_generator[i](ae_in)
                    else:
                        ae_out = self.generator(ae_in)
                    loss += self.l * self.criterion(ae_out.contiguous().view(-1, ae_out.size(-1)), 
                                                ae_y.contiguous().view(-1)) / ae_norm.float()     
            else:
                if self.ae_generator is not None:
                    ae_out = self.ae_generator(ae_x)
                else:
                    ae_out = self.generator(ae_x)
                loss += self.l * self.criterion(ae_out.contiguous().view(-1, ae_out.size(-1)),
                                            ae_y.contiguous().view(-1)) / ae_norm.float()
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float()

def encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, video_features, video_features_mask):
    query_memory, encoded_vid_features, cap_memory, his_memory, ae_encoded_ft = model.encode(query, query_mask, his, his_mask, cap, cap_mask, video_features, video_features_mask)
    return his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft

def greedy_decode(model, batch, max_len, start_symbol, pad_symbol):
    video_features, video_features_mask, cap, cap_mask, his, his_st, his_mask, query, query_mask = batch.fts, batch.fts_mask, batch.cap, batch.cap_mask, batch.his, batch.his_st, batch.his_mask, batch.query, batch.query_mask
    
    his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, video_features, video_features_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(query.data)

    for i in range(max_len-1):
        cap2res_mask = None
        out = model.decode(encoded_vid_features, his_memory, cap_memory, query_memory,
                          video_features_mask, his_mask, cap_mask, query_mask, 
                          Variable(ys), 
                          Variable(subsequent_mask(ys.size(1)).type_as(query.data)),
                          cap2res_mask,
                          ae_encoded_ft)
        if type(out) == list:
            prob = 0
            for idx, o in enumerate(out):
                prob += model.generator[idx](o[:,-1])
        else:
            prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(query.data).fill_(next_word)], dim=1)
    return ys

def beam_search_decode(model, batch, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0, nbest=5, min_len=1):
    video_features, video_features_mask, cap, cap_mask, his, his_st, his_mask, query, query_mask = batch.fts, batch.fts_mask, batch.cap, batch.cap_mask, batch.his, batch.his_st, batch.his_mask, batch.query, batch.query_mask
    
    his_memory, cap_memory, query_memory, encoded_vid_features, ae_encoded_ft = encode(model, his, his_st, his_mask, cap, cap_mask, query, query_mask, video_features, video_features_mask)

    ds = torch.ones(1, 1).fill_(start_symbol).type_as(query.data)
    hyplist=[([], 0., ds)]
    best_state=None
    comp_hyplist=[]
    for l in range(max_len): 
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            cap2res_mask = None
            output = model.decode(encoded_vid_features, his_memory, cap_memory, query_memory,
                                  video_features_mask, his_mask, cap_mask, query_mask,
                                  Variable(st),
                                  Variable(subsequent_mask(st.size(1)).type_as(query.data)),
                                  ae_encoded_ft)
            if type(output) == tuple or type(output) == list:
                logp = model.generator(output[0][:, -1])
            else:
                logp = model.generator(output[:, -1])
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if o == unk_symbol or o == end_symbol:
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).type_as(query.data).fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1,1).type_as(query.data).fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0: 
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None
