#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import chainer
import chainer.links as L
import chainer.functions as F 
 
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
