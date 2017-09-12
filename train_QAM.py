#!/usr/bin/python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import optimizers, cuda
import time
import sys
# import cPickle
import numpy as np
import matplotlib.pyplot as plt
from make_data import *
from sigpy import Make_Step, Quadrature_Amplitude_Modulation

# モデルの宣言
class LSTM(chainer.Chain):
    def __init__(self, in_units=1, hidden_units=2, out_units=1, train=True):
        super(LSTM, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.LSTM(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, out_units),
        )
        self.train = True

    def __call__(self, x, t):
        h = self.l1(x)
        h = self.l2(h)
        y = self.l3(h)
        self.loss = F.mean_squared_error(y, t)
        if self.train:
            return self.loss
        else:
            self.prediction = y
            return self.prediction

    def reset_state(self):
        self.l2.reset_state()

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


# Nは周波数fcの信号の周期を何分割するか
N=64
# 搬送波の周波数
fc = 2
# QAMで信号化したいビット列
letter=['001','100','000','011','101','110']
signal=[]
for i in letter :
   signal.append(int(i,2))

# ここからLSTM
IN_UNITS = 1
HIDDEN_UNITS = 5
OUT_UNITS = 1
TRAINING_EPOCHS = 4000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = len(signal)
LENGTH_OF_SEQUENCE = fc
STEPS_PER_CYCLE = N
NUMBER_OF_CYCLES = fc

xp = cuda.cupy

if __name__ == "__main__":

    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    train_data = data_maker.make(signal)

    plt.title("QAM")
    plt.ylim(min(train_data)-0.2, max(train_data)+0.2)
    plt.plot(range(len(train_data)),train_data)

    plt.show()


    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

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
    cPickle.dump(model, open("./model.pkl", "wb"))

    print("{}[sec]".format(end - start))
