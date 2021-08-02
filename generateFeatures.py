import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import glob

from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

# from qcnn.main_qsr import gen_train_from_wave, labels


windowLength = 2**10
overlapFactor=0.5
windowType='hann'

datasetPath = "./dataset"

datasetFiles = glob.glob(datasetPath + "/**/*.wav", recursive=True)

print(f"Found {len(datasetFiles)} files in the dataset")

