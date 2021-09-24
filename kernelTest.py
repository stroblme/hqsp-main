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


datasetPath = "/storage/mstrobl/dataset"
featurePath = "/storage/mstrobl/features"
checkpointsPath = "/storage/mstrobl/checkpoints"
modelsPath = "/storage/mstrobl/models"
quantumPath = "/storage/mstrobl/dataQuantum"
waveformPath = "/storage/mstrobl/waveforms"
checkpointsPath = "/storage/mstrobl/checkpoints"

exportPath = "/storage/mstrobl/versioning"

TOPIC = "PrepGenTrain"

batchSize = 4
kernelSize = 2
epochs = 30
portion = 2
PoolSize = int(multiprocessing.cpu_count()*0.6) #be gentle..
# PoolSize = 3 #be gentle..

if __name__ == '__main__':
    from stqft.frontend import export

    # export.checkWorkingTree(exportPath)

    import argparse
    parser = argparse.ArgumentParser()
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
    from generateFeatures import gen_features, gen_quantum, reportSettings, samplingRate
    from qcnn.small_qsr import labels
    
    print("Loading from disk...")
    x_train = np.load(f"{featurePath}/x_train_speech.npy")
    x_valid = np.load(f"{featurePath}/x_valid_speech.npy")
    y_train = np.load(f"{featurePath}/y_train_speech.npy")
    y_valid = np.load(f"{featurePath}/y_valid_speech.npy")


    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    print("Loading from disk...")
    q_train_ref = np.load(f"{quantumPath}/quanv_train.npy")
    q_valid_ref = np.load(f"{quantumPath}/quanv_valid.npy")

    q_train, q_valid = gen_quantum(x_train, x_valid, kernelSize, output=None, poolSize=PoolSize, quanv=False)

    assert q_train.shape == q_train_ref.shape
    assert q_valid.shape == q_valid_ref.shape

    # import matplotlib.pyplot as plt

    # import librosa.display
    # plt.figure()
    # for i in range(4):
    #     plt.subplot(5, 1, i+2)
    #     librosa.display.specshow(librosa.power_to_db(q_train[0,:,:,i], ref=np.max))
    #     plt.title('Channel '+str(i+1)+': Quantum Compressed Speech')
    # plt.tight_layout()

    
    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Training @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from fitModel import fit_model

    if args.train:
        # pass quanv data for training and validation
        model, history = fit_model(q_train, y_train, q_valid, y_valid, checkpointsPath)

        data_ix = time.strftime("%Y%m%d_%H%M")
        model.save(f"{modelsPath}/model_{time.time()}")
    else:
        print("Training disabled")

    exp = export(topic=TOPIC, identifier="model", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Model trained (T)/ loaded (F): {args.train}; CheckpointsPath: {checkpointsPath}; ModelsPath: {modelsPath}")
    exp.setData(export.GENERICDATA, {"history_acc":history.history['accuracy'], "history_val_acc":history.history['val_accuracy'], "history_loss":history.history['loss'], "history_val_loss":history.history['val_loss']})
    exp.doExport()