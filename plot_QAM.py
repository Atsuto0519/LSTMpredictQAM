import numpy as np
import matplotlib.pyplot as plt

from sigpy import Make_Step, Quadrature_Amplitude_Modulation

# Nは周波数fcの信号の周期を何分割するか
N=64
t = np.arange(0, N, 1)
# 搬送波の周波数
fc = 1
letter=['001','100','000','011','101','110']
signal=[]
for i in letter :
   signal.append(int(i,2))

# QAM信号の格納用リスト
qam = Quadrature_Amplitude_Modulation(signal, fc, t)

plt.title("QAM")
plt.ylim(min(qam)-0.2, max(qam)+0.2)
plt.plot(range(len(qam)),qam,"o")

plt.show()
