#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import random
from sigpy import Quadrature_Amplitude_Modulation


random.seed(0)

class DataMaker(object):

    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles

    def make(self, signal):
        return np.array(Quadrature_Amplitude_Modulation(signal, 1, np.arange(0, self.steps_per_cycle, 1)))

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = random.randint(0, len(data) - length_of_sequence)
            sequences[i] = data[index:index+length_of_sequence]
        return sequences
