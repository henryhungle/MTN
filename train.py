#!/usr/bin/env python
import argparse
import logging
import math
import sys
import time
import random
import os
import json
import numpy as np
import pickle as pkl
import threading
import pdb 
from tqdm import tqdm 
import torch
import torch.nn as nn
import data_handler as dh
from mtn import *
from label_smoothing import * 
from data_utils import * 

def run_epoch(data, indices, vocab, epoch, model, loss_compute, eval=False):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0 
    total_loss = 0 
    tokens = 0 
    it = tqdm(range(len(indices)), desc="epoch {}/{}".format(epoch+1, args.num_epochs), ncols=0)
    for j in it:
        batch = dh.make_batch(data, indices[j], vocab, separate_caption=args.separate_caption, cut_a=args.cut_a)
        b = batch 
        if True: 
            out, ae_out = model.forward(b)
            if args.auto_encoder_ft == 'caption' or args.auto_encoder_ft == 'summary':
                ntokens_cap = (b.cap != vocab['<blank>']).data.sum()
                loss = loss_compute(out, b.trg_y, b.ntokens, ae_out, b.cap, ntokens_cap)
            elif args.auto_encoder_ft == 'query':
                ntokens_query = (b.query != vocab['<blank>']).data.sum()
                loss = loss_compute(out, b.trg_y, b.ntokens, ae_out, b.query, ntokens_query, is_eval=eval)
        total_loss += loss
        total_tokens += b.ntokens
        tokens += b.ntokens
        if (j+1) % args.report_interval == 0 and not eval:
            elapsed = time.time() - start
            print("Epoch: %d Step: %d Loss: %f Tokens per Sec: %f" %
                    (epoch+1,j+1, loss / b.ntokens.float(), float(tokens) / elapsed))
            with open(train_log_path, "a") as f:
                f.write("{},{},{:e},{}\n".format(epoch+1,j+1,loss/b.ntokens.float(),float(tokens)/elapsed))
            start = time.time()
            tokens = 0
        #prefetch.join()
    return total_loss / total_tokens.float()

##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    # train, dev and test data
    parser.add_argument('--fea-type', nargs='+', type=str, help='Image feature files (.pkl)')
    parser.add_argument('--train-path', default='', type=str,help='Path to training feature files')
    parser.add_argument('--train-set', default='', type=str,help='Filename of train data')
    parser.add_argument('--valid-path', default='', type=str,help='Path to validation feature files')
    parser.add_argument('--valid-set', default='', type=str,help='Filename of validation data')
    parser.add_argument('--include-caption', default='none', type=str, help='Include caption in the history')
    parser.add_argument('--separate-caption', default=0, type=int, help='Separate caption from dialogue history')
    parser.add_argument('--cut-a', default=0, type=int, help='randomly cut responses to simulate bs') 
    parser.add_argument('--merge-source', default=0, type=int, help='merge all source sequences into one') 
    parser.add_argument('--exclude-video', action='store_true',help='')
    parser.add_argument('--fixed-word-emb', default=0, type=int, help='')
    parser.add_argument('--model', default=None, type=str,help='output path of model and params')
    # Model 
    parser.add_argument('--nb-blocks', default=6, type=int,help='number of transformer blocks')
    parser.add_argument('--d-model', default=512, type=int, help='dimension of model tensors') 
    parser.add_argument('--d-ff', default=2048, type=int, help='dimension of feed forward') 
    parser.add_argument('--att-h', default=8, type=int, help='number of attention heads') 
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')  
    parser.add_argument('--separate-his-embed', default=0, type=int, help='Separate the dialog history embedding?')
    parser.add_argument('--separate-cap-embed', default=0, type=int, help='Separate the video caption embedding') 
    parser.add_argument('--diff-encoder', default=0, type=int, help='use different encoder for the autoencoder?') 
    parser.add_argument('--diff-embed', default=0, type=int, help='use different embedding for the autoencoder?') 
    parser.add_argument('--diff-gen', default=0, type=int, help='use different generator for the autoencoder?') 
    parser.add_argument('--auto-encoder-ft', default=None, type=str, help='use what features for autoencoder?')
    # Training 
    parser.add_argument('--num-epochs', '-e', default=15, type=int,help='Number of epochs')
    parser.add_argument('--rand-seed', '-s', default=1, type=int, help="seed for generating random numbers")
    parser.add_argument('--batch-size', '-b', default=32, type=int,help='Batch size in training')
    parser.add_argument('--max-length', default=20, type=int,help='Maximum length for controling batch size')
    parser.add_argument('--max-history-length', default=-1, type=int, help='Maximum past history length to consider')
    parser.add_argument('--report-interval', default=100, type=int,help='report interval to log training results')
    parser.add_argument('--warmup-steps', default=4000, type=int,help='warm up steps for optimizer') 
    parser.add_argument('--loss-l', default=1.0, type=float, help="")
    # others
    parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')
    args = parser.parse_args()
    args.separate_his_embed = bool(args.separate_his_embed)
    args.separate_caption = bool(args.separate_caption)
    args.merge_source = bool(args.merge_source)
    args.separate_cap_embed = bool(args.separate_cap_embed)
    args.cut_a = bool(args.cut_a)
    args.diff_encoder = bool(args.diff_encoder)
    args.diff_embed = bool(args.diff_embed)
    args.diff_gen = bool(args.diff_gen)
    args.fixed_word_emb = bool(args.fixed_word_emb)

    # Presetting
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, 
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(levelname)s: %(message)s')
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    # get vocabulary
    logging.info('Extracting words from ' + args.train_set)
    vocab = dh.get_vocabulary(args.train_set, include_caption=args.include_caption)
    # load data
    logging.info('Loading training data from ' + args.train_set)
    train_data = dh.load(args.fea_type, args.train_path, args.train_set, 
                         include_caption=args.include_caption, separate_caption=args.separate_caption,
                         vocab=vocab, max_history_length=args.max_history_length, 
                         merge_source=args.merge_source)
    logging.info('Loading validation data from ' + args.valid_set)
    valid_data = dh.load(args.fea_type, args.valid_path, args.valid_set, 
                         include_caption=args.include_caption, separate_caption=args.separate_caption, 
                         vocab=vocab, max_history_length=args.max_history_length, 
                         merge_source=args.merge_source)
    if args.fea_type[0] == 'none':
        feature_dims = 0
    else:
        #feature_dims = dh.feature_shape(train_data)
        # TODO: dummy dimension for VisDial image feature; to replace with real dimension
        feature_dims = [2048]
    logging.info("Detected feature dims: {}".format(feature_dims));
    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # make batchset for training
    train_indices, train_samples = dh.make_batch_indices(train_data, args.batch_size,
                                                         max_length=args.max_length, separate_caption=args.separate_caption)
    logging.info('#train sample = %d' % train_samples)
    logging.info('#train batch = %d' % len(train_indices))
    # make batchset for validation
    valid_indices, valid_samples = dh.make_batch_indices(valid_data, args.batch_size,
                                                     max_length=args.max_length, separate_caption=args.separate_caption)
    logging.info('#validation sample = %d' % valid_samples)
    logging.info('#validation batch = %d' % len(valid_indices))
    # create_model
    model = make_model(len(vocab), len(vocab), 
      N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_ff, 
      h=args.att_h, dropout=args.dropout,  
      separate_his_embed=args.separate_his_embed, 
      separate_cap_embed=args.separate_cap_embed, 
      ft_sizes=feature_dims, 
      diff_encoder=args.diff_encoder, 
      diff_embed=args.diff_embed, 
      diff_gen=args.diff_gen,
      auto_encoder_ft=args.auto_encoder_ft) 
    model.cuda()
    criterion = LabelSmoothing(size=len(vocab), padding_idx=vocab['<blank>'], smoothing=0.1)
    criterion.cuda()	
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # save meta parameters
    path = args.model + '.conf'
    with open(path, 'wb') as f:
        pkl.dump((vocab, args), f, -1)
    path2 = args.model + '_params.txt'
    with open(path2, "w") as f: 
        for arg in vars(args):
            f.write("{}={}\n".format(arg, getattr(args, arg)))
 
    logging.info('----------------')
    logging.info('Start training')
    logging.info('----------------')
    # initialize status parameters
    modelext = '.pth.tar'
    min_valid_loss = 1.0e+10
    bestmodel_num = 0
    # save results 
    trace_log_path = args.model+'_trace.csv'
    with open(trace_log_path, "w") as f:
        f.write('epoch,split,avg_loss\n') 
    train_log_path = args.model+'_train.csv'
    with open(train_log_path, "w") as f:  
        f.write('epoch,step,loss,tokens_per_sec\n') 
    print("Saving training results to {}".format(train_log_path))
    print("Saving val results to {}".format(trace_log_path))   
    model_opt = NoamOpt(args.d_model, 1, args.warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(args.num_epochs):
        # start training 
        random.shuffle(train_indices)
        model.train()
        train_loss = run_epoch(train_data, train_indices, vocab, epoch,
                  model,
                  SimpleLossCompute(model.generator, model.auto_encoder_generator,
                  criterion,opt=model_opt, l=args.loss_l))
        logging.info("epoch: %d  train loss: %f" % (epoch+1, train_loss))
        # test on validation data 
        logging.info('-------validation--------')
        model.eval()
        with torch.no_grad():
            valid_loss = run_epoch(valid_data, valid_indices, vocab, epoch,
                    model,
                    SimpleLossCompute(model.generator, model.auto_encoder_generator,
                    criterion,opt=None, l=args.loss_l),
                    eval=True)
        logging.info('epoch: %d validation loss: %f' % (epoch+1, valid_loss))
        with open(trace_log_path,"a") as f:
            f.write("{},train,{:e}\n".format(epoch+1,train_loss))
            f.write("{},val,{:e}\n".format(epoch+1,valid_loss))        
        # update the model and save checkpoints
        modelfile = args.model + '_' + str(epoch + 1) + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)
        if min_valid_loss > valid_loss:
            bestmodel_num = epoch+1
            logging.info('validation loss reduced %.4f -> %.4f' % (min_valid_loss, valid_loss))
            min_valid_loss = valid_loss
            logging.info('a symbolic link is made as ' + args.model + '_best' + modelext)
            if os.path.exists(args.model + '_best' + modelext):
                os.remove(args.model + '_best' + modelext)
            os.symlink(os.path.basename(args.model + '_' + str(bestmodel_num) + modelext), args.model + '_best' + modelext)
        logging.info('----------------')
    logging.info('the best model is epoch %d.' % bestmodel_num)
