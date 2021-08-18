from generateFeatures import gen_features
import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"
print(os.environ.get("LD_LIBRARY_PATH"))
import numpy as np
import time

import multiprocessing
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from generateFeatures import gen_features, gen_quantum
from fitModel import fit_model
from qcnn.small_qsr import labels
from qcnn.models import attrnn_Model

datasetPath = "/ceph/mstrobl/dataset"
featurePath = "/ceph/mstrobl/features"
checkpointsPath = "/ceph/mstrobl/checkpoints"
modelsPath = "/ceph/mstrobl/models"
quantumPath = "/ceph/mstrobl/data_quantum"
waveformPath = "/ceph/mstrobl/waveforms"
checkpointsPath = "/ceph/mstrobl/checkpoints"

samplingRate = 16000
batchSize = 16
epochs = 30
PoolSize = int(multiprocessing.cpu_count()*0.2) #be gentle..

if __name__ == '__main__':
    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Train Time @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    multiprocessing.set_start_method('spawn')
    print(f"Running {PoolSize} processes")

    datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

    print(f"Found {len(datasetFiles)} files in the dataset")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Waveforms @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    x_train, x_valid, y_train, y_valid = gen_features(labels, datasetPath, featurePath, PoolSize, waveformPath=waveformPath, port=10, samplingRate=samplingRate)
    # x_train = np.load(f"{featurePath}/x_train_speech.npy")
    # x_valid = np.load(f"{featurePath}/x_test_speech.npy")
    # y_train = np.load(f"{featurePath}/y_train_speech.npy")
    # y_valid = np.load(f"{featurePath}/y_test_speech.npy")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    q_train, q_valid = gen_quantum(x_train, x_valid, 2, output=quantumPath)

    # q_train = np.load(f"{quantumPath}/quanv_train.npy")
    # q_valid = np.load(f"{quantumPath}/quanv_test.npy")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Training @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    ## For Quanv Exp.
    model = fit_model(q_train, y_train, q_valid, y_valid, checkpointsPath)

    data_ix = time.strftime("%Y%m%d_%H%M")
    model.save(f"{modelsPath}/model_{time.time()}")