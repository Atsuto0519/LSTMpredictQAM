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

random.seed(0)

# Nは周波数fcの信号の周期を何分割するか
N=48
# 搬送波の周波数
fc = 1

# ここからLSTM
MODEL_PATH = "./qam_model.pkl"
IN_UNITS = 1
HIDDEN_UNITS = 5
OUT_UNITS = 1
TRAINING_EPOCHS = 40
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = N
NUMBER_OF_CYCLES = 100

# 誤差計算関数
def compute_loss(model, sequences):
    loss = 0
    rows, cols = sequences.shape
    length_of_sequence = cols
    for i in range(cols - 1):
        x = chainer.Variable(
            xp.asarray(
                [sequences[j, i + 0] for j in range(rows)],
                dtype=np.float32
            )[:, np.newaxis]
        )
        t = chainer.Variable(
            xp.asarray(
                [sequences[j, i + 1] for j in range(rows)],
                dtype=np.float32
            )[:, np.newaxis]
        )
        loss += model(x, t)
    return loss


if __name__ == "__main__":
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # QAMで信号化したいビット列
    signal = []
    for i in range(NUMBER_OF_CYCLES) :
        signal.append(random.random()%8)

    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    train_data = data_maker.make(signal)

    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
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
    for epoch in range(TRAINING_EPOCHS):
        sequences = data_maker.make_mini_batch(train_data, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)
        model.reset_state()
        model.zerograds()
        loss = compute_loss(model, sequences)
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

    end = time.time()

    # save model
    cPickle.dump(model, open("./qam_model.pkl", "wb"))

    print("{}[sec]".format(end - start))
