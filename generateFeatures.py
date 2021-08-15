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

from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

from qcnn.main_qsr import gen_train_from_wave, labels


windowLength = 2**10
overlapFactor=0.875
windowType='hann'

datasetPath = "/ceph/mstrobl/dataset"
waveformPath = "/ceph/mstrobl/waveforms"
featurePath = "/ceph/mstrobl/features/"

av = 0

def gen_mel(speechFile, sr=16000):
    start = time.time()

    y = signal(samplingRate=sr, signalType='file', path=speechFile)
    stqft = transform(stqft_framework, suppressPrint=True, minRotation=0.2, numOfShots=1024)
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType, suppressPrint=True)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', normalize=False, samplingRate=y.samplingRate, nMels=60, fmin=40.0, fmax=y.samplingRate/2)

    diff = time.time()-start
    print(f"Iteration took {diff} s")
    return y_hat_stqft_p

def gen_train(labels, train_audio_path, outputPath, sr=16000, port=1):
    all_wave = list()
    all_label = list()
    
    for label in labels:
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")


        portDatsetLabelFiles = datasetLabelFiles[0::port]
        print(f"Using {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'")

        it = 1
        for datasetLabelFile in portDatsetLabelFiles:
            print(f"Processing '{datasetLabelFile}' in label '{label}' [{it}/{len(portDatsetLabelFiles)}]")
            it+=1

            wave = gen_mel(datasetLabelFile, sr)

            all_wave.append(np.expand_dims(wave[:,1:], axis=2))
            all_label.append(label)

    print(f"Finished generating waveforms at {time.time()}")
    
    with open(f"{waveformPath}/waveforms{time.time()}.pckl", 'wb') as fid:
        pickle.dump(all_wave, fid, pickle.HIGHEST_PROTOCOL)
    with open(f"{waveformPath}/labels{time.time()}.pckl", 'wb') as fid:
        pickle.dump(all_label, fid, pickle.HIGHEST_PROTOCOL)
        
    print(f"Finished dumping cache. Starting Feature export")

    return gen_train_from_wave(all_wave=all_wave, all_label=all_label, output=outputPath)

datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

print(f"Found {len(datasetFiles)} files in the dataset")

gen_train(labels, datasetPath, featurePath, port=10)