import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

from stqft.tests import *

sr=16000

y, _ = librosa.load('/ceph/mstrobl/dataset//left/cb8f8307_nohash_7.wav', sr = sr)
test_plot(y, sr)