#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from numpy.random import *
import math
import random
from sigpy import Quadrature_Amplitude_Modulation


random.seed(0)

class DataMaker(object):

    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles

    def make(self, signal, *, regulary=False, noise=False):
        # データに規則性を付与する
        if (regulary == True) :
            print("regulary:on")
            # 4('100')の後は必ず1('001')が出現するようにsignalを変更
            for i in range(len(signal)) :
                if (signal[i] == 4 and i != len(signal)-1) :
                    signal[i+1] = 1
        else :
            print("regulary:off")


        tmp = np.array(Quadrature_Amplitude_Modulation(signal, 1, np.arange(0, self.steps_per_cycle, 1)))
        res = []
        # 平均0，分散0.05のホワイトノイズを付与
        if (noise == True) :
            print("noise:on")
            for i in tmp :
                res.append(i + normal(0,0.05))
        else :
            print("noise:off")
            res = tmp
        return res

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = random.randint(0, len(data) - length_of_sequence)
            sequences[i] = data[index:index+length_of_sequence]
        return sequences
