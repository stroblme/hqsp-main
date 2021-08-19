import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import time

import multiprocessing
import glob
import numpy as np

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
PoolSize = int(multiprocessing.cpu_count()*0.3) #be gentle..

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveform", type = bool, default = True, action='store_true', help = "Generate Waveforms")
    parser.add_argument("--quantum", type = bool, default = True, action='store_true', help = "Generate Quantum Data")
    parser.add_argument("--train", type = bool, default = True, action='store_true', help = "Fit the model")
    args = parser.parse_args()
    

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
    from generateFeatures import gen_features, gen_quantum
    from qcnn.small_qsr import labels
    
    if args.waveform:
        x_train, x_valid, y_train, y_valid = gen_features(labels, datasetPath, featurePath, PoolSize, waveformPath=waveformPath, port=10, samplingRate=samplingRate)
    else:
        print("Loading from disk...")
        x_train = np.load(f"{featurePath}/x_train_speech.npy")
        x_valid = np.load(f"{featurePath}/x_test_speech.npy")
        y_train = np.load(f"{featurePath}/y_train_speech.npy")
        y_valid = np.load(f"{featurePath}/y_test_speech.npy")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    if args.quantum:
        q_train, q_valid = gen_quantum(x_train, x_valid, 2, output=quantumPath)
    else:
        print("Loading from disk...")
        q_train = np.load(f"{quantumPath}/quanv_train.npy")
        q_valid = np.load(f"{quantumPath}/quanv_test.npy")

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Training @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from fitModel import fit_model

    if args.train:
        ## For Quanv Exp.
        model = fit_model(q_train, y_train, q_valid, y_valid, checkpointsPath)

        data_ix = time.strftime("%Y%m%d_%H%M")
        model.save(f"{modelsPath}/model_{time.time()}")
    else:
        print("Training disabled")