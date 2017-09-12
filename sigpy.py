import numpy as np
import matplotlib.pyplot as plt

def Make_Step(signal, t) :
    res = []
    tmp = []

    for i in range(len(t)):
        tmp.append(1)

    for i in signal:
        res.extend(np.dot(i, tmp))
    return res

def Amplitude_Shift_Keying(signal, fc, t) :
    if(signal == []) :
        return signal

    res = []

    wave0=np.sin(2*np.pi*t/len(t) * fc) * 0
    wave1=np.sin(2*np.pi*t/len(t) * fc)

    for i in signal:
        if i==-1:
            res.extend(wave0)
        else:
            res.extend(wave1)
    return res

def Frequency_Shift_Keying(signal, fc_0, fc_1, t) :
    if(signal == []) :
        return signal

    res = []

    wave0=np.sin(2*np.pi*t/len(t) * fc_0)
    wave1=np.sin(2*np.pi*t/len(t) * fc_1)

    for i in signal:
        if i==-1:
            res.extend(wave0)
        else:
            res.extend(wave1)
    return res

def Phase_Shift_Keying(signal, fc, t) :
    if(signal == []) :
        return signal

    res = []

    wave0=np.cos(2*np.pi*t/len(t) * fc)
    wave1=wave0.copy()*(-1)

    for i in signal:
        if i==1:
            res.extend(wave0)
        else:
            res.extend(wave1)
    return res

def Quadrature_Phase_Shift_Keying(signal, fc, t) :
    if(signal == []) :
        return signal

    res = []

    wave0=np.cos(2*np.pi*t/len(t) * fc)
    wave1=np.cos(2*np.pi*t/len(t) * fc + np.pi/2)
    wave2=np.cos(2*np.pi*t/len(t) * fc + np.pi)
    wave3=np.cos(2*np.pi*t/len(t) * fc + 3*np.pi/2)

    for i in signal:
        if i==0:
            res.extend(wave0)
        elif i==1:
            res.extend(wave1)
        elif i==2:
            res.extend(wave2)
        else:
            res.extend(wave3)
    return res

def Quadrature_Amplitude_Modulation(signal, fc, t) :
    if(signal == []) :
        return signal

    res = []

    wave = []
    wave.append(np.sin(2*np.pi*t/len(t) * fc))
    wave.append(np.sin(2*np.pi*t/len(t) * fc + np.pi/2))
    wave.append(np.sin(2*np.pi*t/len(t) * fc + np.pi))
    wave.append(np.sin(2*np.pi*t/len(t) * fc + 3*np.pi/2))

    for i in signal:
        key = int(i/2)
        res.extend(wave[key]/2 * (i%2+1))

    return res
