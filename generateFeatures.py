from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

from qcnn import main_qsr

frontend.enableInteractive()

windowLength = 2**10
overlapFactor=0.5
windowType='hann'
