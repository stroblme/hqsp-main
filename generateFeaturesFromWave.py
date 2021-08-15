import sys
sys.path.append("./qcnn")

import os
#Activate the cuda env
import glob

import time
import pickle

from qcnn.small_qsr import gen_train_from_wave


datasetPath = "/ceph/mstrobl/dataset"
waveformPath = "/ceph/mstrobl/waveforms"
featurePath = "/ceph/mstrobl/features/"


def gen_train(labels, train_audio_path, outputPath, samplingRate=16000, port=1):
    global sr
    sr = samplingRate
    all_wave = list()

    for label in labels:
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")

        portDatsetLabelFiles = datasetLabelFiles[0::port]
        print(f"\nUsing {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'\n")

    
        with Pool(PoolSize) as p:
            temp_waves = p.map(poolProcess, portDatsetLabelFiles)

        all_wave.append(temp_waves)

    tid = time.time()
    print(f"Finished generating waveforms at {tid}")
    with open(f"{waveformPath}/waveforms{tid}.pckl", 'wb') as fid:
        pickle.dump(all_wave, fid, pickle.HIGHEST_PROTOCOL)
    with open(f"{waveformPath}/labels{tid}.pckl", 'wb') as fid:
        pickle.dump(labels, fid, pickle.HIGHEST_PROTOCOL)
        
    print(f"Finished dumping cache. Starting Feature export")

    return gen_train_from_wave(all_wave=all_wave, all_label=labels, output=outputPath)

if __name__ == '__main__':
    print(f"\n\n\n-----------------------\n\n\n")
    print(f"Generate Feature From Wave @{time.time()}")
    print(f"\n\n\n-----------------------\n\n\n")

    waveformFiles = glob.glob(f"{waveformPath}/waveforms*.pckl")
    waveformFiles.sort(key=os.path.getmtime)
    labelFiles = glob.glob(f"{waveformPath}/labels*.pckl")
    labelFiles.sort(key=os.path.getmtime)

    with open(waveformFiles[-1], 'rb') as fid:
        all_wave = pickle.load(fid)
    with open(labelFiles[-1], 'rb') as fid:
        all_labels = pickle.load(fid)

    # gen_train(labels, datasetPath, featurePath, port=10)
    gen_train_from_wave(all_wave=all_wave, all_label=all_labels, output=featurePath)