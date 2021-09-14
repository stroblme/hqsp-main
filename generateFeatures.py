from qcnn.small_quanv import gen_quanv
import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import glob
import numpy as np

import time
import pickle

# from multiprocessing import Pool
from multiprocessing.dummy import Pool

from stqft.utils import PI
from stqft.frontend import signal, transform
from stqft.stqft import stqft_framework

from qcnn.small_qsr import gen_train_from_wave, gen_train_from_wave_no_split
from qcnn.small_quanv import gen_quanv

nQubits=10
windowLength = 2**nQubits
overlapFactor=0.875
windowType='blackman'

waveformPath = "/ceph/mstrobl/waveforms"

av = 0
sr=16000    #careful: this may be modiefied when calling gen_features

numOfShots=4096
signalFilter=0.02
minRotation=PI/2**(nQubits-6)
nSamplesWindow=1024
overlapFactor=0.875
windowType='blackman'
suppressPrint=True
scale='mel'
normalize=True
nMels=60
fmin=40.0

def reportSettings():
    return f"numOfShots:{numOfShots}; signalFilter:{signalFilter}; minRotation:{minRotation}; nSamplesWindow:{nSamplesWindow}; overlapFactor:{overlapFactor}; windowType:{windowType}; scale:{scale}; normalize:{normalize}; nMels:{nMels}; fmin:{fmin}"

def gen_mel(speechFile):
    print(f"Processing {speechFile}")
    start = time.time()

    #the following parameters are subject of evaluation prior to the training process
    y = signal(samplingRate=sr, signalType='file', path=speechFile)
    stqft = transform(stqft_framework, numOfShots=numOfShots, suppressPrint=suppressPrint, signalFilter=signalFilter, minRotation=minRotation)
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=nSamplesWindow, overlapFactor=overlapFactor, windowType=windowType, suppressPrint=suppressPrint)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=scale, normalize=normalize, samplingRate=y.samplingRate, nMels=nMels, fmin=fmin, fmax=y.samplingRate/2)

    diff = time.time()-start
    print(f"Iteration took {diff} s")
    return y_hat_stqft_p

def poolProcess(datasetLabelFile):
    wave = gen_mel(datasetLabelFile)
    return np.expand_dims(wave[:,1:], axis=2)

def gen_features(labels, train_audio_path, outputPath, PoolSize, waveformPath=None, port=1, split=True):
    all_wave = list()
    all_labels = list()
    
    i=0
    for label in labels:    #iterate over labels, so we don't run into concurrency issues with the mapping
        temp_waves = list()
        
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")  #gather all label specific sample files

        portDatsetLabelFiles = datasetLabelFiles[0::port]   #get only a portion of those files
        print(f"\nUsing {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'\n")

    
        with Pool(PoolSize) as p:
            temp_waves = p.map(poolProcess, portDatsetLabelFiles)   #mapping samples to processes and output back to waveform array

        all_wave = all_wave + temp_waves.copy() #copy to break the reference here
        all_labels = all_labels + [label]*len(portDatsetLabelFiles) #extend the array by the label n times

        print(f"\n Generated {len(temp_waves)} out of {len(port)} with {len(all_labels)} labels")
        print(f"\n---------[Label {i}/{len(labels)}]---------\n")

    tid = time.time()
    print(f"Finished generating waveforms at {tid}")

    if waveformPath != None:
        with open(f"{waveformPath}/waveforms{tid}.pckl", 'wb') as fid:
            pickle.dump(all_wave, fid, pickle.HIGHEST_PROTOCOL)
        with open(f"{waveformPath}/labels{tid}.pckl", 'wb') as fid:
            pickle.dump(all_labels, fid, pickle.HIGHEST_PROTOCOL)
            
        print(f"Finished dumping cache")
    print(f"Starting Feature export")

    #dirty decision, but usefull when called from test.py (where we don't need to split)
    if split:
        return gen_train_from_wave(all_wave=all_wave, all_label=all_labels, output=outputPath)
    else:
        return gen_train_from_wave_no_split(all_wave=all_wave, all_label=all_labels)


def gen_quantum(x_train, x_valid, kr, output):
    #simple pass-through
    return gen_quanv(x_train, x_valid, kr, output)