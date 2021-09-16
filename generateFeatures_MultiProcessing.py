import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import glob

import time

import multiprocessing


from qcnn.small_qsr import labels

from generateFeatures import gen_features

datasetPath = "/ceph/mstrobl/dataset"
waveformPath = "/ceph/mstrobl/waveforms"
featurePath = "/ceph/mstrobl/features/"

PoolSize = int(multiprocessing.cpu_count()*0.6) #be gentle..
av = 0
sr=16000


if __name__ == '__main__':
    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generate Feature Multiprocessing @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    multiprocessing.set_start_method('spawn')
    print(f"Running {PoolSize} processes")

    datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

    print(f"Found {len(datasetFiles)} files in the dataset")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Waveforms @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    
    gen_features(labels, datasetPath, featurePath, PoolSize, waveformPath=waveformPath, portion=10)