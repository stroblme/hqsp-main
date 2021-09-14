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

samplingRate = 16000
batchSize = 16
kernelSize = 2
epochs = 30
port = 1050
# PoolSize = int(multiprocessing.cpu_count()*0.3) #be gentle..
PoolSize = 3 #be gentle..

if __name__ == '__main__':
    from stqft.frontend import export

    export.checkWorkingTree(exportPath)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveform", default = True, action='store_true', help = "Generate Waveforms")
    parser.add_argument("--quantum", default = True, action='store_true', help = "Generate Quantum Data")
    parser.add_argument("--train", default = True, action='store_true', help = "Fit the model")
    args = parser.parse_args()
    

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Train Time @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    multiprocessing.set_start_method('spawn')
    print(f"Running {PoolSize} processes")

    datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

    print(f"Found {len(datasetFiles)} files in the dataset")

    exp = export(topic=TOPIC, identifier="dataset", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Dataset {len(datasetFiles)} in {datasetPath}")
    exp.setData(export.GENERICDATA, datasetFiles)
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Waveforms @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from generateFeatures import gen_features, gen_quantum, reportSettings
    from qcnn.small_qsr import labels
    
    if args.waveform:
        x_train, x_valid, y_train, y_valid = gen_features(labels, datasetPath, featurePath, PoolSize, waveformPath=waveformPath, port=port)
    else:
        print("Loading from disk...")
        x_train = np.load(f"{featurePath}/x_train_speech.npy")
        x_valid = np.load(f"{featurePath}/x_valid_speech.npy")
        y_train = np.load(f"{featurePath}/y_train_speech.npy")
        y_valid = np.load(f"{featurePath}/y_valid_speech.npy")

    exp = export(topic=TOPIC, identifier="waveformData", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Waveforms generated (T)/ loaded (F): {args.waveform}; Labels used: {labels}; FeaturePath: {featurePath}; PoolSize: {PoolSize}; WaveformPath: {waveformPath}; Portioning: {port}, SamplingRate: {samplingRate}, {reportSettings()}")
    exp.setData(export.GENERICDATA, {"x_train":x_train, "x_valid":x_valid, "y_train":y_train, "y_valid":y_valid})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    if args.quantum:
        q_train, q_valid = gen_quantum(x_train, x_valid, kernelSize, output=quantumPath, poolSize=PoolSize)
    else:
        print("Loading from disk...")
        q_train = np.load(f"{quantumPath}/quanv_train.npy")
        q_valid = np.load(f"{quantumPath}/quanv_valid.npy")

    exp = export(topic=TOPIC, identifier="quantumData", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Quantum data generated (T)/ loaded (F): {args.quantum}; FeaturePath: {quantumPath}; PoolSize: {PoolSize};")
    exp.setData(export.GENERICDATA, {"q_train":q_train, "q_valid":q_valid})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Training @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from fitModel import fit_model

    if args.train:
        ## For Quanv Exp.
        model, history = fit_model(q_train, y_train, q_valid, y_valid, checkpointsPath)

        data_ix = time.strftime("%Y%m%d_%H%M")
        model.save(f"{modelsPath}/model_{time.time()}")
    else:
        print("Training disabled")

    exp = export(topic=TOPIC, identifier="model", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Model trained (T)/ loaded (F): {args.train}; CheckpointsPath: {checkpointsPath}; ModelsPath: {modelsPath}")
    exp.setData(export.GENERICDATA, {"history_acc":history.history['acc'], "history_val_acc":history.history['val_acc'], "history_loss":history.history['loss'], "history_val_loss":history.history['val_loss']})
    exp.doExport()