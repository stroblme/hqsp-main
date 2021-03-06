import sys
import time
from numpy import fmax
sys.path.append("./stqft")
sys.path.append("./qcnn")


sr=16000
speechFile = '../dataset/left/cb8f8307_nohash_7.wav'
# speechFile = '/storage/mstrobl/dataset/left/cb8f8307_nohash_7.wav'

if __name__ == '__main__':
    from stqft.tests import *
    from stqft.qft import loadBackend, loadNoiseModel, setupMeasurementFitter
    from generateFeatures import gen_mel, gen_quantum, nQubits, transpOptLvl, numOfShots, numOfRuns, suppressPrint, backend, simulation, signalThreshold, useNoiseModel, noiseMitigationOpt
    from stqft.frontend import frontend
    fri = frontend()

    # y_rosa, _ = librosa.load(speechFile, sr = sr)
    # y_rosa_hat = librosa.feature.melspectrogram(y_rosa, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)

    start = time.time()

    assert simulation
    _, backendInst = loadBackend(backendName=backend, simulation=simulation)
    if useNoiseModel:
        _, noiseModel = loadNoiseModel(backendName=backendInst)

    #     y_hat_stqft_p = gen_mel(audioFile=speechFile, backendInstance=None, noiseModel=None, filterResultCounts=None, show=False)

    if noiseMitigationOpt==1:
        filterResultCounts = setupMeasurementFitter(backendInst, noiseModel,
                                                        transpOptLvl=transpOptLvl, nQubits=nQubits,
                                                        nShots=numOfShots, nRuns=numOfRuns,
                                                        suppressPrint=suppressPrint)

    #     y_hat_stqft_p = gen_mel(audioFile=speechFile, backendInstance=backendInst, noiseModel=noiseModel, filterResultCounts=None, show=False)


    assert noiseMitigationOpt==1
    y_hat_stqft_p = gen_mel(audioFile=speechFile, backendInstance=backendInst, noiseModel=noiseModel, filterResultCounts=filterResultCounts, show=False)

    maxV=0
    for f in y_hat_stqft_p:
        if f.max() > maxV:
            maxV=f.max()

    print(f"Duration: {time.time()-start}")

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
    fri._show(yData=y_hat_stqft_p, x1Data=None, sr = sr, title=f'STQFT_sim_n_mitig, st:{signalThreshold}', xlabel='Time (s)', ylabel='Frequency (Hz)', plotType='librosa')
    fri.primeTime()