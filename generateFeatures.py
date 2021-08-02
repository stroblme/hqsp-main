import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
import glob

os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

# from qcnn.main_qsr import gen_train_from_wave, labels

def genFeature(speechFile):
    y = signal(samplingRate=16000, signalType='file', path=speechFile)
    stqft = transform(stqft_framework, suppressPrint=True, minRotation=0.2)
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', fmax=4000)


windowLength = 2**10
overlapFactor=0.5
windowType='hann'

datasetPath = "/ceph/mstrobl/dataset"
labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

print(f"Found {len(datasetFiles)} files in the dataset")

for label in labels:
    glob.glob(datasetPath + "/**/*.wav", recursive=True)

    all_wave = list()
    all_label = list()

    wave = genFeature(dataFile)

