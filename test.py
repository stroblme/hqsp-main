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


datasetPath = "/storage/mstrobl/testDataset"
featurePath = "/storage/mstrobl/testFeatures"
modelsPath = "/storage/mstrobl/models"
quantumPath = "/storage/mstrobl/testDataQuantum"
waveformPath = "/storage/mstrobl/testWaveforms"

TESTMODEL="model_1635183947.5163796"

exportPath = "/storage/mstrobl/versioning"

TOPIC = "PrepGenTest"

batchSize = 28
kernelSize = 2
epochs = 40
portion = 1
PoolSize = int(multiprocessing.cpu_count()*0.6) #be gentle..
# PoolSize = 1 #be gentle..

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveform", default = 1, help = "Generate Waveforms")
    parser.add_argument("--quantum", default= 1, help = "Generate Quantum Data")
    parser.add_argument("--checkTree", default = 1, help = "Checks if the working tree is dirty")
    args = parser.parse_args()


    from stqft.frontend import export

    if int(args.checkTree) == 1:
        export.checkWorkingTree(exportPath)
    

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Test Time @{time.time()}")
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
    from generateFeatures import gen_features, gen_quantum, reportSettings, samplingRate
    from qcnn.small_qsr import labels
    
    if int(args.waveform)==1:
        x_test, y_test = gen_features(labels, datasetPath, featurePath, PoolSize, waveformPath=waveformPath, portion=portion, split=False)
    else:
        print("Loading from disk...")
        x_test = np.load(f"{featurePath}/x_test_speech.npy")
        y_test = np.load(f"{featurePath}/y_test_speech.npy")

    exp = export(topic=TOPIC, identifier="waveformData", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Waveforms generated (T)/ loaded (F): {args.waveform}; Labels used: {labels}; FeaturePath: {featurePath}; PoolSize: {PoolSize}; WaveformPath: {waveformPath}; Portioning: {portion}, SamplingRate: {samplingRate}, {reportSettings()}")
    exp.setData(export.GENERICDATA, {"x_test":x_test, "y_test":y_test})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generating Quantum Data @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    # disable quanv and pix chan mal
    if int(args.quantum)==-2:
        q_test = x_test
    # enable quanv
    elif int(args.quantum)==1:
        q_test, _ = gen_quantum(x_test, [], kernelSize, output=quantumPath, poolSize=PoolSize)
    # pix chan map
    elif int(args.quantum)==-1:
        q_test, _ = gen_quantum(x_test, [], kernelSize, output=quantumPath, poolSize=PoolSize, quanv=False)
    # load from disk
    else:
        print("Loading from disk...")
        q_test = np.load(f"{quantumPath}/quanv_test.npy")

    exp = export(topic=TOPIC, identifier="quantumData", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"Quantum data generated (T)/ loaded (F): {args.quantum}; FeaturePath: {quantumPath}; PoolSize: {PoolSize};")
    exp.setData(export.GENERICDATA, {"q_test":q_test})
    exp.doExport()

    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Starting Testing @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")
    from fitModel import evaluate_model

    # model = get_model(f"{modelsPath}/model_{time.time()}")
    #if quanv completely disabled and no pix channel map
    # if int(args.quantum)==-2 or q_test.shape[3]==1:
    #     print("using ablation")
    #     # pass quanv data for training and validation
    #     model, history = model.evaluate(q_test, y_test, epochs=epochs, batchSize=batchSize, ablation=True)
    # else:
        # pass quanv data for training and validation
    result = evaluate_model(q_test, y_test, f"{modelsPath}/{TESTMODEL}", epochs=epochs, batchSize=batchSize)

    exp = export(topic=TOPIC, identifier="model", dataDir=exportPath)
    exp.setData(export.DESCRIPTION, f"ModelsPath: {modelsPath}, Model: {TESTMODEL}")
    exp.setData(export.GENERICDATA, {"result":result})
    exp.doExport()