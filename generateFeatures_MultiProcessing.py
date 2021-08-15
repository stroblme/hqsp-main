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

import multiprocessing
from multiprocessing import Pool


from stqft.frontend import signal, transform
from stqft.stqft import stqft_framework

from qcnn.small_qsr import gen_train_from_wave, labels


windowLength = 2**10
overlapFactor=0.875
windowType='hann'

datasetPath = "/ceph/mstrobl/dataset"
waveformPath = "/ceph/mstrobl/waveforms"
featurePath = "/ceph/mstrobl/features/"

PoolSize = int(multiprocessing.cpu_count()*0.6) #be gentle..
av = 0
sr=16000

def gen_mel(speechFile):
    print(f"Processing {speechFile}")
    start = time.time()

    y = signal(samplingRate=sr, signalType='file', path=speechFile)
    stqft = transform(stqft_framework, suppressPrint=True, minRotation=0.2, numOfShots=1024)
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType, suppressPrint=True)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', normalize=False, samplingRate=y.samplingRate, nMels=60, fmin=40.0, fmax=y.samplingRate/2)

    diff = time.time()-start
    print(f"Iteration took {diff} s")
    return y_hat_stqft_p

def poolProcess(datasetLabelFile):
    wave = gen_mel(datasetLabelFile)
    return np.expand_dims(wave[:,1:], axis=2)

def gen_train(labels, train_audio_path, outputPath, samplingRate=16000, port=1):
    global sr
    sr = samplingRate
    all_wave = list()
    all_labels = list()
    
    for label in labels:
        temp_waves = list()
        
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")

        portDatsetLabelFiles = datasetLabelFiles[0::port]
        print(f"\nUsing {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'\n")

    
        with Pool(PoolSize) as p:
            temp_waves = p.map(poolProcess, portDatsetLabelFiles)

        all_wave = all_wave + temp_waves.copy() #copy to break the reference here
        all_labels = all_labels + [label]*len(portDatsetLabelFiles) #append the label n times

    tid = time.time()
    print(f"Finished generating waveforms at {tid}")
    with open(f"{waveformPath}/waveforms{tid}.pckl", 'wb') as fid:
        pickle.dump(all_wave, fid, pickle.HIGHEST_PROTOCOL)
    with open(f"{waveformPath}/labels{tid}.pckl", 'wb') as fid:
        pickle.dump(all_labels, fid, pickle.HIGHEST_PROTOCOL)
        
    print(f"Finished dumping cache. Starting Feature export")

    return gen_train_from_wave(all_wave=all_wave, all_label=all_labels, output=outputPath)

if __name__ == '__main__':
    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generate Feature Multiprocessing @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    multiprocessing.set_start_method('spawn')
    print(f"Running {PoolSize} processes")

    datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

    print(f"Found {len(datasetFiles)} files in the dataset")

    gen_train(labels, datasetPath, featurePath, port=10)