import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import glob
import numpy as np

import time

import multiprocessing

samplingRate = 16000
windowLength = 2**10
overlapFactor=0.875
windowType='hann'
port=10

testDatasetPath = "/ceph/mstrobl/testDataset"
waveformPath = "/ceph/mstrobl/testWaveforms"
featurePath = "/ceph/mstrobl/testFeatures"
quantumPath = "/ceph/mstrobl/testDataQuantum"

modelsPath = "/ceph/mstrobl/models"

exportPath = "/ceph/mstrobl/versioning"

TOPIC = "PrepGenTest"


PoolSize = int(multiprocessing.cpu_count()*0.2) #be gentle..

if __name__ == '__main__':
    from stqft.frontend import export

    export.checkWorkingTree(exportPath)

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Test Time @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    multiprocessing.set_start_method('spawn')
    print(f"Running {PoolSize} processes")

    datasetFiles = glob.glob(testDatasetPath + "/**/*.wav", recursive=True)
    print(f"Found {len(datasetFiles)} files in the dataset")

    exp = export(topic=TOPIC, identifier="dataset", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Dataset {len(datasetFiles)} in {testDatasetPath}")
    exp.setData(export.GENERICDATA, datasetFiles)
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Waveforms @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from generateFeatures import gen_features, reportSettings
    from qcnn.small_qsr import labels

    x, y = gen_features(labels, testDatasetPath, featurePath, PoolSize, port=port, split=False, samplingRate=samplingRate) # use 10 samples

    exp = export(topic=TOPIC, identifier="waveforms", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Labels used: {labels}; FeaturePath: {featurePath}; PoolSize: {PoolSize}; WaveformPath: {waveformPath}; Portioning: {port}, SamplingRate: {samplingRate}, {reportSettings()}")
    exp.setData(export.GENERICDATA, {"x":x, "y":y})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from qcnn.small_quanv import gen_qspeech

    q = gen_qspeech(x, [], 2) 

    exp = export(topic=TOPIC, identifier="quantumData", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Quantum data; FeaturePath: {quantumPath}")
    exp.setData(export.GENERICDATA, {"q":q})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Loading Model @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from tensorflow.keras.models import load_model

    models = sorted(glob.glob(f"{modelsPath}/**"), key = os.path.getmtime)
    print(f"Using model {models[-1]}")
    model = load_model(models[-1], compile = True)


    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Predictions @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    y_preds = np.argmax(model.predict(q), axis=1)

    errors = dict.fromkeys(labels, 0)

    for idx in range(0, y.shape[0]-1):
        y_idx = np.argmax(y[idx], axis=0)

        print(f"Model returned {labels[y_preds[idx]]} and label was {labels[y_idx]}")

        if labels[y_preds[idx]] != labels[y_idx]:
            errors[labels[y_idx]] += 1

    print(f"Made {errors} errors in {y.size} samples")

    print(f"\n\n\n-----------------------\n\n\n")

    print(f"Error distribution over labels\n")
    for label, nErrors in errors.items():
        print(f"{label}:\t" + "+"*nErrors)

    exp = export(topic=TOPIC, identifier="errors", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Errors of {y.shape[0]} samples in model; ModelsPath:{modelsPath}; Labels:{labels}")
    exp.setData(export.GENERICDATA, errors)
    exp.doExport()