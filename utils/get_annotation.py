#!/usr/bin/python
# coding: utf-8

import argparse
import json
import sys
import re

from stopword_filter import StopwordFilter


parser = argparse.ArgumentParser()
parser.add_argument('--stopwords', '-s', default='', type=str,
                    help='read a stopword list from file')
parser.add_argument('--dictmap', '-d', default='', type=str,
                    help='dictmap file (.json)')
parser.add_argument('--last', '-l', action='store_true',
                    help='extract only last answers')
parser.add_argument('dialog_file', 
                    help='dialog data file (.json)')
parser.add_argument('output_file', 
                    help='output file (.json)')
args = parser.parse_args()

# read stop words
if args.stopwords:
    swfilter = StopwordFilter(args.stopwords)
else:
    swfilter = None
# read dict map
if args.dictmap:
    dictmap=json.load(open(args.dictmap,'r'))
else:
    dictmap = None

# output data structure
data = {}
data['info'] = {}
data['licenses'] = []
data['type'] = 'captions'

annos = []
images = []
image_id=1
cap_id=1

# store data
dialogs=json.load(open(args.dialog_file,'r'))
for dialog in dialogs['dialogs']:
    vid = dialog['image_id']
    if  dictmap is not None:
        vid = dictmap[vid]
    for n, qa in enumerate(dialog['dialog']):
        if args.last==False or n==len(dialog['dialog'])-1:
            idx = '%s_%d' % (vid, n)
            sent = dialog['dialog'][n]['answer']
            if swfilter:
                sent = swfilter(sent.encode('utf-8'))
            annos.append({"image_id": cap_id, "id": cap_id, "caption": sent})
            images.append({"name": idx, "id": cap_id})
            cap_id += 1
        
data['annotations'] = annos
data['images'] = images

# output file
json.dump(data, open(args.output_file, 'w'), indent=4)
