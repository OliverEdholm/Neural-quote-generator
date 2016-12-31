#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Quote LSTM

Neural network with LSTM to generate quotes.

Oliver Edholm, 14 years old 2016-12-30 14:13
'''
# imports
import os
import logging
from shutil import rmtree
from six.moves import urllib
from six.moves import xrange
from six.moves import cPickle as pickle

import tflearn
from tflearn.data_utils import random_sequence_from_textfile
from tflearn.data_utils import textfile_to_semi_redundant_sequences

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
QUOTES_URL = 'https://raw.githubusercontent.com/alvations/Quotables/master/author-quote.txt'
DOWNLOADED_QUOTES_PATH = 'quotes.txt'

CHAR_IDX_PATH = 'charidx.pkl'

CHECKPOINTS_DIR_PATH = 'checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, 'quote_model')

N_ITERATIONS = 50
LOWERCASE_QUOTES = True
MAXLEN = 30


def get_txt_file(file_path):
    logging.debug('getting text file at {}'.format(file_path))
    with open(file_path, 'r') as txt_file:
        return txt_file.read()


def save_pkl_file(data, file_path):
    logging.debug('saving data at {}'.format(file_path))
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def download_quotes():
    def download_raw_quotes():
        logging.debug('downloading raw quotes from {}'.format(QUOTES_URL))
        urllib.request.urlretrieve(QUOTES_URL, DOWNLOADED_QUOTES_PATH)

    def process_raw_quotes():
        logging.debug('processing raw quotes')
        raw_quotes = get_txt_file(DOWNLOADED_QUOTES_PATH).splitlines()
        
        with open(DOWNLOADED_QUOTES_PATH, 'w') as txt_file:
            txt_file.flush()
            
            for line in raw_quotes:
                quote = line.split('\t')[1]
                if LOWERCASE_QUOTES:
                    quote = quote.lower()
                txt_file.write(quote + '\n\n')

    logging.info('downloading quotes')
    download_raw_quotes()
    process_raw_quotes()


def build_model(char_idx):
    logging.info('building model')
    model = tflearn.input_data([None, MAXLEN, len(char_idx)])
    
    n_lstm_neurons = 512
    dropout = 0.4
    for _ in xrange(2):
        model = tflearn.lstm(model, n_lstm_neurons, return_seq=True)
        model = tflearn.dropout(model, dropout)

    for _ in xrange(1):
        model = tflearn.lstm(model, n_lstm_neurons)
        model = tflearn.dropout(model, dropout)

    model = tflearn.fully_connected(model, len(char_idx),
                                    activation='softmax')
    model = tflearn.regression(model, optimizer='adam',
                               loss='categorical_crossentropy',
                               learning_rate=0.001)

    return tflearn.SequenceGenerator(model, dictionary=char_idx,
                                     seq_maxlen=MAXLEN, clip_gradients=5.0,
                                     checkpoint_path=CHECKPOINT_PATH)


def generate_quote(gen, seed, length, temperature):
    print('-- Quote with temperature {} --'.format(temperature))
    print(gen.generate(length, temperature=temperature, seq_seed=seed))


def train_model(gen, X, Y, char_idx):
    if os.path.exists(CHECKPOINTS_DIR_PATH):
        yes_or_no = input('There already exists a checkpoint folder, do you \
wish to replace it? y/n: ')
        if yes_or_no[0].lower() == 'y':
            logging.debug('removing folder {}'.format(CHECKPOINTS_DIR_PATH))
            rmtree(CHECKPOINTS_DIR_PATH)
        else:
            if not yes_or_no[0].lower() == 'n':
                print('unknown input')
            logging.info('terminating program')
            return

    logging.debug('creating folder {}'.format(CHECKPOINTS_DIR_PATH))
    os.makedirs(CHECKPOINTS_DIR_PATH)

    logging.info('training model')
    for _ in xrange(N_ITERATIONS):
        seed = random_sequence_from_textfile(DOWNLOADED_QUOTES_PATH, MAXLEN)
        gen.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1,
                run_id='quote')
        
        print()
        print('-- TESTING --')
        generate_quote(gen, seed, 100, temperature)
        generate_quote(gen, seed, 100, temperature)


def main():
    if not os.path.isfile(DOWNLOADED_QUOTES_PATH):
        download_quotes()

    logging.info('creating training data')
    X, Y, char_idx = textfile_to_semi_redundant_sequences(DOWNLOADED_QUOTES_PATH,
                                                          seq_maxlen=MAXLEN)
    save_pkl_file(char_idx, CHAR_IDX_PATH)

    gen = build_model(char_idx)
    
    train_model(gen, X, Y, char_idx)


if __name__ == '__main__':
    main()

