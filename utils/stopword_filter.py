import os
import re

class StopwordFilter(object):

    def __init__(self, filename):
        self.pats = []
        if os.path.exists(filename):
            for ln in open(filename, 'r').readlines():
                ww = ln.split()
                if len(ww)==1:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ''))
                elif len(ww)==2:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ww[1]))

    def _filter(self, input_words):
        output_words = []
        for w in input_words:
            target = w
            for p in self.pats:
                v = p[0].sub(p[1],w)
                if v != w:
                    target = v
                    break
            if target != '':
                output_words.append(target)
        return output_words

    def __call__(self, input_words):
        if isinstance(input_words, str):
            return ' '.join(self._filter(input_words.split()))
        elif isinstance(input_words, list):
            return self._filter(input_words)
        else:
            return None
