import multiprocessing
from qcnn.small_quanv import gen_quanv
import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import glob
import numpy as np

import time
import pickle

# from multiprocessing import Pool
from multiprocessing import Pool

from stqft.utils import PI
from stqft.frontend import signal, transform
from stqft.stqft import stqft_framework
from stqft.qft import loadBackend, loadNoiseModel, setupMeasurementFitter

from qcnn.small_qsr import gen_train_from_wave, gen_train_from_wave_no_split
from qcnn.small_quanv import gen_quanv


av = 0

nQubits=10
samplingRate=16000    #careful: this may be modified when calling gen_features
numOfShots=4096
signalThreshold=0.02 #optimized according to thesis
minRotation=0.2 #PI/2**(nQubits-4)
overlapFactor=0.875
windowLength = 2**nQubits
windowType='blackman'
suppressPrint=True
useNoiseModel=True
backend="ibmq_guadalupe" #ibmq_guadalupe, ibmq_melbourne (noisier)
noiseMitigationOpt=1
numOfRuns=1
simulation=True
transpileOnce=True
transpOptLvl=1
fixZeroSignal=False
scale='mel'
normalize=True
nMels=60
fmin=40.0
enableQuanv=True


def reportSettings():
    return f"numOfShots:{numOfShots}; signalFilter:{signalThreshold}; minRotation:{minRotation}; nSamplesWindow:{nSamplesWindow}; overlapFactor:{overlapFactor}; windowType:{windowType}; scale:{scale}; normalize:{normalize}; nMels:{nMels}; fmin:{fmin}"

def gen_mel(audioFile:str, backendInstance=backend, noiseModel=None, filterResultCounts=None, show=False, minRotation=minRotation):
    global backendStorage

    print(f"Processing {audioFile}")
    start = time.time()

    #the following parameters are subject of evaluation prior to the training process
    # Frontend Signal instantiation
    y = signal(samplingRate=samplingRate, signalType='file', path=audioFile)
    # QFT init
    stqft = transform(stqft_framework, 
                        numOfShots=numOfShots, 
                        minRotation=minRotation, signalThreshold=signalThreshold, fixZeroSignal=fixZeroSignal,
                        suppressPrint=suppressPrint, draw=False,
                        simulation=simulation,
                        noiseMitigationOpt=noiseMitigationOpt, filterResultCounts=filterResultCounts,
                        useNoiseModel=useNoiseModel, noiseModel=noiseModel, backend=backendInstance, 
                        transpileOnce=transpileOnce, transpOptLvl=transpOptLvl)

    # STQFT init
    y_hat_stqft, f, t = stqft.forward(y, 
                            nSamplesWindow=windowLength,
                            overlapFactor=overlapFactor,
                            windowType=windowType,
                            suppressPrint=suppressPrint)
    # Frontend Post Processing
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=scale, normalize=normalize, samplingRate=y.samplingRate, nMels=nMels, fmin=fmin, fmax=y.samplingRate/2)

    diff = time.time()-start
    print(f"Iteration took {diff} s")

    if show:
        stqft.show(y_hat_stqft_p, f_p, t_p, title=f"STQFT")
    return y_hat_stqft_p

def poolProcess(labelFileAndBackendInstance:list):
    wave = gen_mel(*labelFileAndBackendInstance)
    return np.expand_dims(wave[:,1:], axis=2)

def gen_features(labels:list, train_audio_path:str, outputPath:str, PoolSize:int, waveformPath:str=None, portion:int=1, split:bool=True):
    all_wave = list()
    all_labels = list()

    # need to do some pre-initialization mostly because of api restrictions and resources concerns
    _, backendInstance = loadBackend(backendName=backend, simulation=simulation)
    _, noiseModel = loadNoiseModel(backendName=backendInstance)

    if noiseMitigationOpt != 0:
        filterResultCounts = setupMeasurementFitter(backendInstance, noiseModel,
                                                    transpOptLvl=transpOptLvl, nQubits=nQubits,
                                                    nShots=numOfShots, nRuns=numOfRuns,
                                                    suppressPrint=suppressPrint)
    else:
        filterResultCounts = None

    for i, label in enumerate(labels):    #iterate over labels, so we don't run into concurrency issues with the mapping
        print(f"\n---------[Label {i}/{len(labels)}]---------\n")
        temp_waves = list()
        
        datasetLabelFiles = glob.glob(f"{train_audio_path}/{label}/*.wav")  #gather all label specific sample files

        # TODO: maybe change that to "random"?
        portDatsetLabelFiles = datasetLabelFiles[0::portion]   #get only a portion of those files
        # ^ (validated) ^
        print(f"\nUsing {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'\n")

        with Pool(PoolSize) as p:
            temp_waves = p.map(poolProcess, list(zip(portDatsetLabelFiles,[backendInstance]*len(portDatsetLabelFiles), [noiseModel]*len(portDatsetLabelFiles), [filterResultCounts]*len(portDatsetLabelFiles))))   #mapping samples to processes and output back to waveform array
        # ^ (validated) ^ When running "single threaded" in the multiprocessing.dummy module with PoolSize=1
            # ^ (validated) ^ When running in standard multiprocessing module with PoolSize=3


        #appending waves and labels at the END of both arrays 
        all_wave = all_wave + temp_waves.copy() #copy to break the reference here
        # ^ (validated) ^
        all_labels = all_labels + [label]*len(portDatsetLabelFiles) #extend the array by the label n times
        # ^ (validated) ^

        print(f"\n Generated {len(temp_waves)} waves. In total {len(all_wave)} waves and {len(all_labels)} labels\n")

    tid = time.time()
    print(f"Finished generating waveforms at {tid}")

    if waveformPath != None:
        with open(f"{waveformPath}/waveforms.pckl", 'wb') as fid:
            pickle.dump(all_wave, fid, pickle.HIGHEST_PROTOCOL)
        with open(f"{waveformPath}/labels.pckl", 'wb') as fid:
            pickle.dump(all_labels, fid, pickle.HIGHEST_PROTOCOL)
            
        print(f"Finished dumping cache")
    print(f"Starting Feature export")

    #dirty decision, but usefull when called from test.py (where we don't need to split)
    if split:
        return gen_train_from_wave(all_wave=all_wave, all_label=all_labels, output=outputPath)
    else:
        return gen_train_from_wave_no_split(all_wave=all_wave, all_label=all_labels)


def gen_quantum(x_train, x_valid, kr, output, poolSize=1, quanv=enableQuanv):
    #simple pass-through
    return gen_quanv(x_train, x_valid, kr, output, poolSize, quanv=quanv)
