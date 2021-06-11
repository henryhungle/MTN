#!/usr/bin/env python

import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle
import json

import numpy as np
import six

import torch
import torch.nn as nn
import data_handler as dh
import pdb
from data_utils import * 

# Evaluation routine
def generate_response(model, data, batch_indices, vocab, maxlen=20, beam=5, penalty=2.0, nbest=1, ref_data=None):
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        question_set = data['original']['data']['questions']
        answer_set = data['original']['data']['answers']
        for idx, dialog in enumerate(data['original']['data']['dialogs']):
            vid = dialog['image_id']
            if args.undisclosed_only:
                out_dialog = dialog['dialog'][-1:]
            else:
                out_dialog = dialog['dialog']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(out_dialog):
                #if args.undisclosed_only:
                #    assert qa['answer'] == '__UNDISCLOSED__'
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + question_set[qa['question']])
                #if args.undisclosed_only and ref_data is not None:
                #    logging.info('REF: ' + ref_dialog[t]['answer'])
                #else:
                #logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                batch, answer_options = dh.make_batch(data, batch_indices[qa_id], vocab, separate_caption=train_args.separate_caption, is_test=True)
                qa_id += 1
                logs = []
                for answer_option in answer_options:
                    log = get_log(answer_option, model, batch)
                    logs.append(log) 
                ranked_logs = np.argsort(logs)[::-1]
                answer_option_indices = qa['answer_options']
                for n in range(min(nbest, len(logs))):
                    pred_idx = ranked_logs[n]
                    hypstr = answer_set[answer_option_indices[pred_idx]]
                    logging.info('HYP[%d]: %s' % (n + 1, hypstr))
                    if n == 0: 
                        pred_dialog['dialog'][t]['answer'] = answer_option_indices[pred_idx]
                        pred_dialog['dialog'][t]['answer_pred_idx'] = pred_idx 
                        pred_dialog['dialog'][t]['answer_str'] = hypstr  

    return {'dialogs': result_dialogs}


##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test-path', default='', type=str,
                        help='Path to test feature files')
    parser.add_argument('--test-set', default='', type=str,
                        help='Filename of test data')
    parser.add_argument('--model-conf', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--maxlen', default=30, type=int,
                        help='Max-length of output sequence')
    parser.add_argument('--beam', default=3, type=int,
                        help='Beam width')
    parser.add_argument('--penalty', default=2.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Number of n-best hypotheses')
    parser.add_argument('--output', '-o', default='', type=str,
                        help='Output generated responses in a json file')
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')
    parser.add_argument('--decode-style', default='greedy', type=str, help='greedy or beam_search')
    parser.add_argument('--undisclosed-only', default=0, type=int, help='')
    parser.add_argument('--labeled-test', default=None, type=str, help='directory to labelled data')

    args = parser.parse_args()
    args.undisclosed_only = bool(args.undisclosed_only)
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model)
    path = args.model_conf
    with open(path, 'rb') as f:
        vocab, train_args = pickle.load(f)
    model = torch.load(args.model+'.pth.tar')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # prepare test data
    logging.info('Loading test data from ' + args.test_set)
    test_data = dh.load(train_args.fea_type, args.test_path, args.test_set,
                        vocab=vocab, 
                        include_caption=train_args.include_caption, separate_caption=train_args.separate_caption,
                        max_history_length=train_args.max_history_length,
                        merge_source=train_args.merge_source,
                        undisclosed_only=args.undisclosed_only, 
                        is_test=True)
    test_indices, test_samples = dh.make_batch_indices(test_data, 1, separate_caption=train_args.separate_caption)
    logging.info('#test sample = %d' % test_samples)
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    labeled_test = None 
    if args.undisclosed_only and args.labeled_test is not None:
        labeled_test = json.load(open(args.labeled_test, 'r'))
    result = generate_response(model, test_data, test_indices, vocab, 
                               maxlen=args.maxlen, beam=args.beam, 
                               penalty=args.penalty, nbest=args.nbest, ref_data=labeled_test)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
