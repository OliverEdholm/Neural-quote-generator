#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Evaluate LSTM

Program for evaluating checkpoint created by quote_lstm.py.

Oliver Edholm, 14 years old 2016-12-31 06:43
'''
# imports
import sys
import logging
from six.moves import cPickle as pickle

import quote_lstm

import tflearn
from tflearn.data_utils import random_sequence_from_textfile

# setup
logging.basicConfig(level=logging.DEBUG)


# functions
def get_pkl_file(file_path):
    logging.debug('getting pickle file at {}'.format(file_path))
    with open(file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def main():
    checkpoint_path = sys.argv[1]

    char_idx = get_pkl_file(quote_lstm.CHAR_IDX_PATH)
    
    gen = quote_lstm.build_model(char_idx)

    logging.debug('loading model')
    gen.load(checkpoint_path)

    while True:
        print()
        temperature = float(input('Which temperature do you want: '))
        length = int(input('What length do you want: '))
        print('\n'*100)

        seed = random_sequence_from_textfile(quote_lstm.DOWNLOADED_QUOTES_PATH,
                                             quote_lstm.MAXLEN)
        print()
        print('-'*100)
        quote_lstm.generate_quote(gen, seed, length, temperature)
        print('-'*100)

        print('Press CTRL+C if you want to quit this program.')
        print()


if __name__ == '__main__':
    main()

