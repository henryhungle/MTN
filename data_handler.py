#!/usr/bin/env python

import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import pdb 
import torch 
from data_utils import * 

def get_npy_shape(filename):
    # read npy file header and return its shape
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape

def align_vocab(pretrained_vocab, vocab, pretrained_weights):
    for module, module_wt in pretrained_weights.items():
        for layer, layer_wt in module_wt.items():
            if 'embed' in layer:
                print("Aligning word emb for layer {} in module {}...".format(layer, module))
                print("Pretrained emb of shape {}".format(layer_wt.shape))
                emb_dim = layer_wt.shape[1]
                embs = np.zeros((len(vocab), emb_dim), dtype=np.float32)
                count = 0 
                for k,v in vocab.items():
                    if k in pretrained_vocab:
                        embs[v] = layer_wt[pretrained_vocab[k]]
                    else:
                        count += 1 
                pretrained_weights[module][layer] = embs
                print("Aligned emb of shape {}".format(embs.shape))
                print("Number of unmatched words {}".format(count))
    return pretrained_weights

def get_vocabulary(dataset_file, cutoff=1, include_caption='none'):
    vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = {}
    for dialog in dialog_data['data']['dialogs']:
        if include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary':
            if include_caption == 'caption' or include_caption == 'summary':
                caption = dialog[include_caption]
            else:
                caption = dialog['caption'] + dialog['summary']
            for word in caption.split():
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
    for key in ['questions', 'answers']:
        for sent in dialog_data['data'][key]:
            for word in sent.split():
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
    cutoffs = [1,2,3,4,5]
    for cutoff in cutoffs:
        vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3}
        for word, freq in word_freq.items():
            if freq > cutoff:
                vocab[word] = len(vocab) 
        print("{} words for cutoff {}".format(len(vocab), cutoff))
    return vocab

def words2ids(str_in, vocab):
    words = str_in.split()
    sentence = np.ndarray(len(words)+2, dtype=np.int32)
    sentence[0]=vocab['<sos>']
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i+1] = vocab[w]
        else:
            sentence[i+1] = vocab['<unk>']
    sentence[-1]=vocab['<eos>']
    return sentence

# Load text data
def load(fea_types, fea_path, dataset_file, vocab, include_caption='none', separate_caption=False, max_history_length=-1, merge_source=False, undisclosed_only=False, is_test=False):
    dialog_data = json.load(open(dataset_file, 'r'))
    dialog_list = []
    vid_set = set()
    qa_id = 0
    question_set = dialog_data['data']['questions']
    answer_set = dialog_data['data']['answers']
    for dialog in dialog_data['data']['dialogs']:
        if include_caption == 'caption' or include_caption == 'summary':
            caption = words2ids(dialog[include_caption], vocab)
        elif include_caption == 'caption,summary':
            caption = words2ids(dialog['caption'] + dialog['summary'], vocab)
        else:
            caption = np.array([vocab['<blank>']], dtype=np.int32)
        questions = [words2ids(question_set[d['question']], vocab) for d in dialog['dialog']]
        if not is_test:
            answers = [words2ids(answer_set[d['answer']], vocab) for d in dialog['dialog']]
        else:
            answers = [words2ids(answer_set[d['answer']], vocab) for d in dialog['dialog'][:-1]]
            answers.append(np.asarray([vocab['<sos>']]))
            answer_options = [words2ids(answer_set[a], vocab) for a in dialog['dialog'][-1]['answer_options']]
        qa_pair = [np.concatenate((q,a)).astype(np.int32) for q,a in zip(questions, answers)]
        vid = dialog['image_id']
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions)-1,len(questions))
        else:
            it = range(len(questions))
        for n in it:
            #if undisclosed_only:
            #    assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                history = [np.array([vocab['<blank>']], dtype=np.int32)]
            else:
                history = [caption]
            if max_history_length > 0: 
                start_turn_idx = max(0, n - max_history_length)
            else:
                start_turn_idx = 0 
            for m in range(start_turn_idx, n):
                history = np.append(history, qa_pair[m])
            if type(history) == list: #only including caption i.e. no dialogue history 
                history = history[0]
            question = questions[n]
            if merge_source:
                question = np.concatenate((caption, history, question))
            answer_in = answers[n][:-1]
            answer_out = answers[n][1:]
            item = [vid, qa_id, history, question, answer_in, answer_out]
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                item.append(caption)
            if is_test:
                item.append(answer_options)
            dialog_list.append(item)
            qa_id += 1
    data = {'dialogs': dialog_list, 'vocab': vocab, 'features': [], 
            'original': dialog_data}
    if fea_types is not None and fea_types[0] != 'none':
        for ftype in fea_types:
            basepath = fea_path.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filepath = basepath.replace('<ImageID>', str(vid))
                #shape = get_npy_shape(filepath)
                shape = [100] 
                # TODO: dummy shape for VisDial image feature; replace with real shape 
                features[vid] = (filepath, shape[0])
            data['features'].append(features)
    else:
        data['features'] = None 
    return data 

def make_batch_indices(data, batchsize=100, max_length=20, separate_caption=False):
    # Setup mini-batches
    idxlist = []
    for n, dialog in enumerate(data['dialogs']):
        vid = dialog[0]  # video ID
        x_len = []
        if data['features'] is not None:
            for feat in data['features']:
                value = feat[vid]
                size = value[1] if isinstance(value, tuple) else len(value)
                x_len.append(size)
        else:
            x_len.append(0)
        qa_id = dialog[1]  # QA-pair id
        h_len = len(dialog[2]) # history length
        q_len = len(dialog[3]) # question length
        a_len = len(dialog[4]) # answer length
        if separate_caption:
            c_len = len(dialog[6])
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len, c_len))
        else:
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len))
    if batchsize > 1:
        if separate_caption:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[6],-s[2][0],-s[4],-s[5]))
        else:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[2][0],-s[4],-s[5]))
    n_samples = len(idxlist)
    batch_indices = []
    bs = 0
    while bs < n_samples:
        in_len = idxlist[bs][3]
        bsize = int(batchsize / int(in_len / max_length + 1))
        be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
        #pdb.set_trace()
        x_len = [ max(idxlist[bs:be], key=lambda s:s[2][j])[2][j]
                for j in six.moves.range(len(x_len))]
        h_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
        q_len = max(idxlist[bs:be], key=lambda s:s[4])[4]
        a_len = max(idxlist[bs:be], key=lambda s:s[5])[5]
        if separate_caption:
            c_len = max(idxlist[bs:be], key=lambda s:s[6])[6]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]
        # index[0]: video ids 
        # index[1]: question-answer ids 
        # index[2]: length of video frame sequences for each feature type
        # index[3]: max length of the dialogue history 
        # index[4]: max length of questions
        # index[5]: max length of answers
        # index[-1]: number of dialogues 
        if separate_caption:
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, c_len, be - bs))
        else:
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, be - bs))
        bs = be
    return batch_indices, n_samples

def pad_seq(seqs, max_length, pad_token):
  output = []
  for seq in seqs:
    result = np.ones(max_length, dtype=seq.dtype)*pad_token
    result[:seq.shape[0]] = seq 
    output.append(result)
  return output 

def prepare_data(seqs):
  return torch.from_numpy(np.asarray(seqs)).cuda().long()

def make_batch(data, index, vocab, separate_caption=False, skip=[1,1,1], cut_a=False, cut_a_p=0.5, is_test=False):
    if separate_caption:
        x_len, h_len, q_len, a_len, c_len, n_seqs = index[2:]
    else:
        x_len, h_len, q_len, a_len, n_seqs = index[2:]
    if data['features'] is not None:
        feature_info = data['features']
    else:
        feature_info = [] 
    for j in six.moves.range(n_seqs):
        if len(feature_info) == 0: 
            x_batch = None
            continue 
        vid = index[0][j]
        #fea = [np.load(fi[vid][0])[::skip[idx]] for idx,fi in enumerate(feature_info)]
        fea = [np.random.rand(100, 2048) for ft in feature_info]
        # TODO: dummy feature for VisDial feature; to replace with real feature 
        if j == 0:
            # pad the video features with ones to the max #seq in the batch
            x_batch = [np.ones((x_len[i], n_seqs, fea[i].shape[-1]),dtype=np.float32)
              if len(fea[i].shape)==2 else np.zeros((x_len[i], n_seqs, fea[i].shape[-2], fea[i].shape[-1]),dtype=np.float32)
              for i in six.moves.range(len(x_len))]

        for i in six.moves.range(len(feature_info)):
            x_batch[i][:len(fea[i]), j] = fea[i]
    pad = vocab['<blank>']
    h_batch = []
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    c_batch = None
    if separate_caption:
        c_batch = [] 
    h_st_batch = None
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        history, question, answer_in, answer_out = dialogs[qa_id][2:6]
        if cut_a:
            pr = np.random.uniform()
            if pr >= (1-cut_a_p):
                end_idx = np.random.choice(range(1, len(answer_in)), 1)[0]
                answer_out = np.concatenate((answer_in[1:end_idx],[answer_in[end_idx]]))
                answer_in = answer_in[:end_idx]
        if separate_caption:
            c_batch.append(dialogs[qa_id][6])
        h_batch.append(history)
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)
    h_batch = prepare_data(pad_seq(h_batch, h_len, pad))
    q_batch = prepare_data(pad_seq(q_batch, q_len, pad))
    a_batch_in = prepare_data(pad_seq(a_batch_in, a_len, pad))
    a_batch_out = prepare_data(pad_seq(a_batch_out, a_len, pad))
    if separate_caption:
      c_batch = prepare_data(pad_seq(c_batch, c_len, pad))
    batch = Batch(q_batch, h_batch, h_st_batch, x_batch, c_batch, a_batch_in, a_batch_out, pad)
    if is_test:
        return batch, dialogs[qa_id][7]
    return batch 


def feature_shape(data):
    dims = []
    for features in data["features"]:
        sample_feature = list(features.values())[0]
        if isinstance(sample_feature, tuple):
	        dims.append(np.load(sample_feature[0]).shape[-1])
        else:
            dims.append(sample_feature.shape[-1])
    return dims
