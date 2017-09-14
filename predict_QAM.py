#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
import pickle
import numpy as np
from chainer import optimizers, cuda
import argparse
import chainer
import random
from make_data import *

random.seed(0)

MODEL_PATH = "./qam_model.pkl"
PREDICTION_LENGTH = 75
PREDICTION_PATH = "./prediction_5.txt"
INITIAL_PATH = "./initial.txt"
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100
 
 
def predict_sequence(model, input_seq, output_seq, dummy):
    sequences_col = len(input_seq)
    model.reset_state()
    for i in range(sequences_col):
        x = chainer.Variable(xp.asarray(input_seq[i:i+1], dtype=np.float32)[:, np.newaxis])
        future = model(x, dummy)
    cpu_future = chainer.cuda.to_cpu(future.data)
    return cpu_future
 
 
def predict(seq, model, pre_length, initial_path, prediction_path):
    # initial sequence 
    input_seq = np.array(seq[:seq.shape[0]/4])
 
    output_seq = np.empty(0)
 
    # append an initial value
    output_seq = np.append(output_seq, input_seq[-1])
 
    model.train = False
    dummy = chainer.Variable(xp.asarray([0], dtype=np.float32)[:, np.newaxis])
 
    for i in range(pre_length):
        future = predict_sequence(model, input_seq, output_seq, dummy)
        input_seq = np.delete(input_seq, 0)
        input_seq = np.append(input_seq, future)
        output_seq = np.append(output_seq, future)
 
    with open(prediction_path, "w") as f:
        for (i, v) in enumerate(output_seq.tolist(), start=input_seq.shape[0]):
            f.write("{i} {v}\n".format(i=i-1, v=v))
 
    with open(initial_path, "w") as f:
        for (i, v) in enumerate(seq.tolist()):
            f.write("{i} {v}\n".format(i=i, v=v))
 
 
if __name__ == "__main__":
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    with open(MODEL_PATH, mode='rb') as f :
        model = pickle.load(f)

    # cuda環境では以下のようにすればよい
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    signal = []
    for i in range(NUMBER_OF_CYCLES) :
        signal.append(random.random()%8)
 
    # make data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    data = data_maker.make(signal)
    sequences = data_maker.make_mini_batch(data, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)
 
    sample_index = 45
    predict(sequences[sample_index], model, PREDICTION_LENGTH, INITIAL_PATH, PREDICTION_PATH)