"""Data handler to load and prepare train/test batches.

Author(s): Hung Le, Satwik Kottur
"""
#!/usr/bin/env python

import collections
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

from nltk.tokenize import word_tokenize

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


def get_vocabulary(dataset_file, cutoff=1, predict_belief_states=False):
    """Create vocabulary by excluding words below threshold.
    """
    SPECIAL_TOKENS = {
        '<unk>': 0,
        '<blank>': 1,
        '<user>': 2,
        "<system>": 3,
        '<eos>': 4,
        "<belief>": 5,
    }
    vocab = copy.deepcopy(SPECIAL_TOKENS)
    dialog_data = json.load(open(dataset_file, 'r'))
    word_freq = collections.Counter()
    for dialog_datum in dialog_data["dialogue_data"]:
        for turn_datum in dialog_datum["dialogue"]:
            for key in ["transcript", "system_transcript"]:
                for word in word_tokenize(turn_datum[key]):
                    word_freq[word] += 1

    # If belief state is to be predicted, add tokens to vocab.
    if predict_belief_states:
        for dialog_datum in dialog_data["dialogue_data"]:
            for turn_datum in dialog_datum["dialogue"]:
                user_belief = turn_datum["transcript_annotated"]
                str_belief_state_per_frame = (
                    "{act} [ {slot_values} ] ( {request_slots} ) < >".format(
                        act=user_belief["act"].strip(),
                        slot_values=', '.join(
                            ['{} = {}'.format(k.strip(), str(v).strip())
                                for k, v in user_belief['act_attributes']['slot_values'].items()]),
                        request_slots=', '.join(user_belief['act_attributes']['request_slots']),
                    )
                )
                # Add addition tokens to vocabulary.
                for token in str_belief_state_per_frame.split():
                    word_freq[token] += 1

    cutoffs = [1, 2, 3, 4, 5]
    for cutoff in cutoffs:
        vocab = copy.deepcopy(SPECIAL_TOKENS)
        for word, freq in word_freq.items():
            if freq > cutoff:
                vocab[word] = len(vocab)
        print("{} words for cutoff {}".format(len(vocab), cutoff))
    return vocab


def words2ids(str_in, vocab, speaker=None):
    # Use NLTK to tokenize.
    if speaker == "belief":
        words = str_in.split()
    else:
        words = word_tokenize(str_in)
    sentence = np.ndarray(len(words) + 2, dtype=np.int32)
    # assert speaker is not None, "Speaker must be non-empty!"
    SPEAKER_MAP = {"user": "<user>", "system": "<system>", "belief": "<belief>"}
    sentence[0] = vocab[SPEAKER_MAP.get(speaker, "<blank>")]
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i+1] = vocab[w]
        else:
            sentence[i+1] = vocab['<unk>']
    sentence[-1]=vocab['<eos>']
    return sentence


def get_image_feature_key(scene_label):
    """Get image feature key given the scene labels.
    """
    key = "{}.png".format(scene_label)
    if key[:2] == "m_":
        key = key[2:]
    return key


# Load text data
def load(
    fea_types, fea_path, dataset_file, vocab, max_history_length=-1,
    merge_source=False, undisclosed_only=False, is_test=False,
    predict_belief_states=False
):
    dialog_data = json.load(open(dataset_file, 'r'))
    dialog_list = []
    vid_set = set()
    qa_id = 0
    for dialog_datum in dialog_data["dialogue_data"]:
        user_utterances = [
            words2ids(ii["transcript"], vocab, "user")
            for ii in dialog_datum["dialogue"]
        ]

        if predict_belief_states:
            belief_states = []
            for turn_datum in dialog_datum["dialogue"]:
                user_belief = turn_datum["transcript_annotated"]
                str_belief_state_per_frame = (
                    "{act} [ {slot_values} ] ( {request_slots} ) < >".format(
                        act=user_belief["act"].strip(),
                        slot_values=', '.join(
                            ['{} = {}'.format(k.strip(), str(v).strip())
                                for k, v in user_belief['act_attributes']['slot_values'].items()]),
                        request_slots=', '.join(user_belief['act_attributes']['request_slots']),
                    )
                )
                belief_states.append(
                    words2ids(str_belief_state_per_frame, vocab, "belief")
                )

            # Add multimodal objects using previous turn system annotations.
            # multimodal_context = [words2ids("<SOM> <EOM>", vocab)]
            # vocab["<SOM>"] = len(vocab)
            # vocab["<MOM>"] = len(vocab)
            # for turn_datum in dialog_datum["dialogue"][:-1]:
            #     system_transcript = turn_datum["system_transcript_annotated"]
            #     object_ids = system_transcript["act_attributes"]["objects"]
            #     multimodal_str = words2ids(
            #         "<SOM> {} <MOM>".format(
            #             ", ".join([str(oo) for oo in object_ids])
            #         ), vocab
            #     )
            #     multimodal_context.append(multimodal_str)

        if not is_test:
            system_utterances = [
                words2ids(ii["system_transcript"], vocab, "system")
                for ii in dialog_datum["dialogue"]
            ]
        else:
            system_utterances = [
                words2ids(ii["system_transcript"], vocab, "system")
                for ii in dialog_datum["dialogue"][:-1]
            ]
            system_utterances.append(np.asarray([vocab["<system>"]]))
        utterance_pairs = [
            np.concatenate((user, system)).astype(np.int32)
            for user, system in zip(user_utterances, system_utterances)
        ]
        vid = tuple(dialog_datum["scene_ids"].values())
        vid_set.add(vid)

        if undisclosed_only:
            it = range(len(user_utterances) - 1, len(user_utterances))
        else:
            it = range(len(user_utterances))
        for n in it:
            #if undisclosed_only:
            #    assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            history = np.array([vocab['<blank>']], dtype=np.int32)
            if max_history_length > 0:
                start_turn_idx = max(0, n - max_history_length)
            else:
                start_turn_idx = 0
            for m in range(start_turn_idx, n):
                if predict_belief_states:
                    history = np.append(history, utterance_pairs[m])
                else:
                    history = np.append(history, utterance_pairs[m])
            user_utterance = user_utterances[n]
            if merge_source:
                user_utterance = np.concatenate((history, user_utterance))
            if not predict_belief_states:
                system_in = system_utterances[n][:-1]
                system_out = system_utterances[n][1:]
            else:
                system_in = belief_states[n][:-1]
                system_out = belief_states[n][1:]
            item = [vid, qa_id, history, user_utterance, system_in, system_out]
            if is_test:
                item.append([])
            dialog_list.append(item)
            qa_id += 1
    data = {
        'dialogs': dialog_list, 'vocab': vocab, 'features': [], 'original': dialog_data
    }
    print("# dialogs: {}".format(len(dialog_list)))
    if fea_types is not None and fea_types[0] != 'none':
        print("Reading features: {}".format(fea_path))
        with open(fea_path, "r") as file_id:
            image_features = pickle.load(file_id)["cummulative_embed"]
        features = {}
        for vid in vid_set:
            vid_features = []
            for scene_vid in vid:
                key = get_image_feature_key(scene_vid)
                if key not in image_features:
                    print("Image features not found: {}".format(key))
                    vid_features.append(np.zeros((10, 516)))
                else:
                    vid_features.append(image_features[key])
            # Image features hardcoded.
            features[vid] = (np.concatenate(vid_features, axis=0), 516)
        data['features'] = [features]
    else:
        data['features'] = None
    return data


def make_batch_indices(data, batchsize=100, max_length=20):
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
        idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len))
    if batchsize > 1:
        idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[2][0],-s[4],-s[5]))
    n_samples = len(idxlist)
    batch_indices = []
    bs = 0
    while bs < n_samples:
        in_len = idxlist[bs][3]
        bsize = int(batchsize / int(in_len / max_length + 1))
        be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
        x_len = [ max(idxlist[bs:be], key=lambda s:s[2][j])[2][j]
                for j in six.moves.range(len(x_len))]
        h_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
        q_len = max(idxlist[bs:be], key=lambda s:s[4])[4]
        a_len = max(idxlist[bs:be], key=lambda s:s[5])[5]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]
        # index[0]: video ids 
        # index[1]: question-answer ids 
        # index[2]: length of video frame sequences for each feature type
        # index[3]: max length of the dialogue history 
        # index[4]: max length of questions
        # index[5]: max length of answers
        # index[-1]: number of dialogues 
        batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, be - bs))
        bs = be
    return batch_indices, n_samples


def pad_seq(seqs, max_length, pad_token):
  output = []
  for seq in seqs:
    result = np.ones(max_length, dtype=seq.dtype) * pad_token
    result[:seq.shape[0]] = seq
    output.append(result)
  return output


def prepare_data(seqs):
  return torch.from_numpy(np.asarray(seqs)).cuda().long()


def make_batch(data, index, vocab, skip=[1,1,1], cut_a=False, cut_a_p=0.5, is_test=False):
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
        fea = [fi[vid][0] for idx, fi in enumerate(feature_info)]
        if j == 0:
            # pad the video features with ones to the max #seq in the batch
            x_batch = [
                np.ones((x_len[i], n_seqs, fea[i].shape[-1]), dtype=np.float32)
                    if len(fea[i].shape) == 2
                    else np.zeros((x_len[i], n_seqs, fea[i].shape[-2], fea[i].shape[-1]), dtype=np.float32)
              for i in six.moves.range(len(x_len))
            ]

        for i in six.moves.range(len(feature_info)):
            x_batch[i][:len(fea[i]), j] = fea[i]
    pad = vocab['<blank>']
    h_batch = []
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    c_batch = None
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
        h_batch.append(history)
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)
    h_batch = prepare_data(pad_seq(h_batch, h_len, pad))
    q_batch = prepare_data(pad_seq(q_batch, q_len, pad))
    a_batch_in = prepare_data(pad_seq(a_batch_in, a_len, pad))
    a_batch_out = prepare_data(pad_seq(a_batch_out, a_len, pad))
    batch = Batch(q_batch, h_batch, h_st_batch, x_batch, c_batch, a_batch_in, a_batch_out, pad)
    # if is_test:
    #     return batch, dialogs[qa_id][7]
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
