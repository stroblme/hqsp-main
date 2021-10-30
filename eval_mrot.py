import sys
import time
from numpy import fmax
sys.path.append("./stqft")
from stqft.utils import PI
sys.path.append("./qcnn")


sr=16000
speechFile = '../dataset/left/cb8f8307_nohash_7.wav'
# speechFile = '/storage/mstrobl/dataset/left/cb8f8307_nohash_7.wav'

if __name__ == '__main__':
    # from stqft.tests import *
    from stqft.qft import loadBackend, loadNoiseModel, setupMeasurementFitter
    # from generateFeatures import gen_mel, gen_quantum, nQubits, transpOptLvl, numOfShots, numOfRuns, suppressPrint, backend, simulation, signalThreshold, useNoiseModel, noiseMitigationOpt
    from generateFeatures import gen_mel
    from stqft.frontend import frontend, export
    nQubits=10
    samplingRate=16000    #careful: this may be modified when calling gen_features
    numOfShots=4096
    signalThreshold=0.06 #optimized according to thesis
    minRotation=0.2 #PI/2**(nQubits-4)
    overlapFactor=0.875
    windowLength = 2**nQubits
    windowType='blackman'
    suppressPrint=True
    useNoiseModel=True
    backend="ibmq_guadalupe" #ibmq_guadalupe, ibmq_melbourne (noisier)
    noiseMitigationOpt=0
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

    TOPIC = "speech_sim_n_mrot"
    export.checkWorkingTree("/media/veracrypt1/QuantumMachineLearningDevelopment/main/stqft/data")
    fri = frontend()

    # y_rosa, _ = librosa.load(speechFile, sr = sr)
    # y_rosa_hat = librosa.feature.melspectrogram(y_rosa, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)

    assert simulation
    _, backendInst = loadBackend(backendName=backend, simulation=simulation)
    assert useNoiseModel
    _, noiseModel = loadNoiseModel(backendName=backendInst)
    
    pt=1
    mrot=PI/2**(nQubits-pt)
    while mrot <= PI/2:

        ylabel = "Frequency (Hz)" if pt == 1 or pt == 4 or pt == 7 else " "
        xlabel = "Time (s)" if pt >= 7 else " "

        assert noiseMitigationOpt==0
        y_hat_stqft_p = gen_mel(audioFile=speechFile, backendInstance=backendInst, noiseModel=noiseModel, filterResultCounts=None, show=False, minRotation=mrot,signalThreshold=signalThreshold,noiseMitigationOpt=noiseMitigationOpt)

        plotData = fri._show(yData=y_hat_stqft_p, subplot=[3,nQubits/2-2,pt], x1Data=None, sr = sr, title=f'STQFT_sim_n, mr:{mrot:.3f}', ylabel=ylabel, xlabel=xlabel, plotType='librosa', xticks=[0, 1, 2, 3, 4])


        exp = export(topic=TOPIC, identifier=f"stqft_sim_n_mr_{mrot:.2f}", dataDir="/media/veracrypt1/QuantumMachineLearningDevelopment/main/stqft/data")
        exp.setData(export.SIGNAL, y_hat_stqft_p)
        exp.setData(export.DESCRIPTION, f"stqft, {backend}, chirp, window: 'blackman', length=2**10, mrot:{mrot}")
        exp.setData(export.PLOTDATA, plotData)
        exp.doExport()

        pt += 1
        mrot = PI/2**(nQubits-pt)

    # q_train, q_valid = gen_quantum([y_hat_stqft_p], [], 2, output="./", poolSize=1, quanv=True)

    # y = signal(samplingRate=sr, signalType='file', path=speechFile)

    # stqft = transform(stqft_framework, numOfShots=2048, suppressPrint=True, signalFilter=True)
    # y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=1024, overlapFactor=0.875, windowType='hamm')
    # y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', normalize=True, samplingRate=y.samplingRate, nMels=60, fmin=40.0, fmax=y.samplingRate/2)
    # y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='none', normalize=False)

    # mel_basis = librosa.filters.mel(sr, f.size, n_mels=60, fmin=40.0, fmax=sr/2)

    # y_hat_stqft_p_mel = np.dot(mel_basis[:,1:], y_hat_stqft_p)

    # fri._show(yData=y_rosa_hat, x1Data=None, sr = sr, title='STFT_sim', xlabel='Time (s)', ylabel='Frequency (Hz)', plotType='librosa')
    # fri._show(yData=y_hat_stqft_p, x1Data=None, sr = sr, title=f'STQFT_sim, st:{signalThreshold}', xlabel='Time (s)', ylabel='Frequency (Hz)', plotType='librosa')
    # fri._show(yData=y_hat_stqft_p, x1Data=None, sr = sr, title=f'STQFT_sim_n, st:{signalThreshold}', xlabel='Time (s)', ylabel='Frequency (Hz)', plotType='librosa')
    fri.primeTime()