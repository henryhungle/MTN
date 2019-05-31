#!/usr/bin/python

import argparse
import json
import sys
import re

from stopword_filter import StopwordFilter


parser = argparse.ArgumentParser()
parser.add_argument('--stopwords', '-s', default='', type=str,
                    help='read a stopword list from file')
parser.add_argument('--last', '-l', action='store_true',
                    help='store only last answers')
parser.add_argument('result_file', help='dialog result file (.json)')
parser.add_argument('hypout_file', help='output hypothesis file (.json)')
args = parser.parse_args()

# read stop words
if args.stopwords:
    swfilter = StopwordFilter(args.stopwords)
else:
    swfilter = None

annos = []
result = json.load(open(args.result_file, 'r'))
image_id=1
for dialog in result['dialogs']:
    vid = dialog['image_id']
    for n, qa in enumerate(dialog['dialog']):
        if args.last==False or n==len(dialog['dialog'])-1:
            idx = '%s_%d' % (vid, n)
            sent = dialog['dialog'][n]['answer']
            if swfilter:
                sent = swfilter(sent.encode('utf-8'))
            annos.append({'image_id': image_id, 'caption': sent})
            image_id += 1
        
json.dump(annos, open(args.hypout_file,'w'), indent=4)
