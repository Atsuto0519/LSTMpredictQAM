#!/usr/bin/python
# -*- coding: utf-8 -*-

from make_data import *
from lstm import *
import numpy as np
from chainer import optimizers, cuda
import argparse
import random
import time
import sys
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from sigpy import Make_Step, Quadrature_Amplitude_Modulation
import glob


random.seed(0)

# Nは周波数fcの信号の周期を何分割するか
N=50
# 搬送波の周波数
fc = 1

# ここからLSTM
MODEL_PATH = "./train_model_RNN_vertical.pkl"
# IN_UNITS = 1
HIDDEN_UNITS = 5
OUT_UNITS = 1
TRAINING_EPOCHS = 10000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 50
LENGTH_OF_SEQUENCE = 50
STEPS_PER_CYCLE = N
NUMBER_OF_CYCLES = 50
TRAIN_ERROR = "train_error_for.txt"

# 誤差計算関数
def compute_loss(model, sequences):
    loss = 0
    rows, cols = sequences.shape
    length_of_sequence = cols

    X = chainer.Variable(np.delete(sequences,rows-1,0))
    tmp = np.delete(sequences,0,0)
    tmp = np.asarray([[i[460]] for i in tmp])
    T = chainer.Variable(tmp)
    return model(X, T)


if __name__ == "__main__":
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)

    # load npy data
    save = '../../../raw_data/data*.npy'
    files = glob.glob(save)
    files.sort()

    train_data = []
    for f in files :
        var = np.load(f)
        train_data.append([i[450] for i in var])

    # check model's out units
    vertical_size = var.shape[2]

    # setup model
    model = RNN(vertical_size, HIDDEN_UNITS, OUT_UNITS)
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    # cuda環境では以下のようにすればよい
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    start = time.time()
    cur_start = start

    # 0番目はテストデータ
    num_test = 0

    list_train = np.delete(np.arange(len(train_data)), num_test)
    rand_list = np.array([np.random.permutation(list_train) for x in range(TRAINING_EPOCHS)])
    f = open(TRAIN_ERROR, "w")

    for epoch in range(TRAINING_EPOCHS) :
        sequences = np.array(train_data[rand_list[int(epoch/3)][epoch%3]])
        model.reset_state()
        model.zerograds()
        loss = compute_loss(model, sequences)
        # loss = model(sequences, sequences)
        loss.backward()
        optimizer.update()

        if epoch != 0 and epoch % DISPLAY_EPOCH == 0:
            cur_end = time.time()
            # display loss
            print(
                "[{j}]training loss:\t{i}\t{k}[sec/epoch]".format(
                    j=epoch,
                    i=loss.data/(sequences.shape[1] - 1),
                    k=(cur_end - cur_start)/DISPLAY_EPOCH
                )
            )
            cur_start = time.time()
            sys.stdout.flush()
            f.write("{j} {i}\n".format(j=epoch,i=loss.data/(sequences.shape[1] - 1),))

    end = time.time()

    # save model
    cPickle.dump(model, open(MODEL_PATH, "wb"))

    print("{}[sec]".format(end - start))
