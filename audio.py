import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from time import time as now
from multiprocessing import Process, Queue

fs = 8000

def brown_noise(blocksize=1024, leak_rate=1.0, fs=44100, amplitude=1.0):
    maxval = np.iinfo(np.int16).max
    h = 1.0 / fs
    a = np.array([1.0, - (1.0 - h * leak_rate)], dtype='float32')
    b = np.array([h, 0.0], dtype='float32')
    y0 = np.array([0.0], dtype='float32')
    zi = signal.lfiltic(b, a, y0)
    yield
    while True:
        random = (np.random.rand(blocksize).astype('float32') - 0.5) * fs ** (1/2)
        c = random * amplitude  
        y, zi = signal.lfilter(b, a, c, zi=zi)
        if any(y>1):
            print('over')
        if any(y<-1):
            print('under')
        y[y>1.0] = 2.0 - y[y>1.0]
        y[y<-1.0] = -2.0 - y[y<-1.0]
        numsamples = yield (y[:,None] * maxval).astype('int16')
           

blocksize=1024
noise_gen = brown_noise(blocksize=blocksize, fs=fs, amplitude=5, leak_rate=10)
noise_gen.send(None)


def callback(outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = q.get_nowait() 

def produce(q):
    while True:
        q.put(noise_gen.send(0))
    
q = Queue(maxsize=2)
p = Process(target=produce, args=(q,))
p.start()
with sd.OutputStream(latency='high', blocksize=blocksize, dtype='int16', samplerate=fs, callback=callback, channels=1):
    while True:
        sd.sleep(100)
