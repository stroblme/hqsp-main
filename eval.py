import sys

from numpy import fmax
sys.path.append("./stqft")
sys.path.append("./qcnn")

from stqft.tests import *
from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

sr=16000
speechFile = '../dataset/left/cb8f8307_nohash_7.wav'

y_rosa, _ = librosa.load(speechFile, sr = sr)
y_rosa_hat = librosa.feature.melspectrogram(y_rosa, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
# y_rosa_hat = librosa.feature.melspectrogram(y_rosa, sr=sr, n_fft=1024, hop_length=128, power=1.0)

y = signal(samplingRate=sr, signalType='file', path=speechFile)

stqft = transform(stqft_framework, numOfShots=2048, suppressPrint=True, signalFilter=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=1024, overlapFactor=0.875, windowType='hamm')
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', normalize=True, samplingRate=y.samplingRate, nMels=60, fmin=40.0, fmax=y.samplingRate/2)
# y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='none', normalize=False)

# mel_basis = librosa.filters.mel(sr, f.size, n_mels=60, fmin=40.0, fmax=sr/2)

# y_hat_stqft_p_mel = np.dot(mel_basis[:,1:], y_hat_stqft_p)

# test_plot(y_rosa_hat, sr)
test_plot(y_hat_stqft_p, sr)

input()