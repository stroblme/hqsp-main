import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import glob
import numpy as np


from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

from qcnn.main_qsr import gen_train_from_wave, labels

windowLength = 2**10
overlapFactor=0.5
windowType='hann'

datasetPath = "/ceph/mstrobl/dataset"
outputPath = "/ceph/mstrobl/features/"

def gen_mel(speechFile, sr=16000):
    y = signal(samplingRate=sr, signalType='file', path=speechFile)
    stqft = transform(stqft_framework, suppressPrint=True, minRotation=0.2, numOfShots=1024)
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType, suppressPrint=True)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', fmax=4000)

    return y_hat_stqft_p

def gen_train(labels, train_audio_path, outputPath, sr=16000, port=1):
    for label in labels:
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")

        all_wave = list()
        all_label = list()

        portDatsetLabelFiles = datasetLabelFiles[0::port]
        print(f"Using {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'")

        it = 0
        for datasetLabelFile in portDatsetLabelFiles:
            print(f"Processing '{datasetLabelFile}' in label '{label}' [{it}/{len(portDatsetLabelFiles)}]")
            it+=1

            wave = gen_mel(datasetLabelFile, sr)

            all_wave.append(np.expand_dims(wave, axis=2))
            all_label.append(label)

    print(f"Finished generating waveforms. Start exporting features")

    return gen_train_from_wave(all_wave=all_wave, all_label=all_label, output=outputPath)

datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

print(f"Found {len(datasetFiles)} files in the dataset")

gen_train(labels, datasetPath, outputPath, port=2)