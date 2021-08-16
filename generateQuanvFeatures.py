import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"
print(os.environ.get("LD_LIBRARY_PATH"))
import numpy as np

from qcnn.small_quanv import gen_quanv

datasetPath = "/ceph/mstrobl/dataset"
featurePath = "/ceph/mstrobl/features"
checkpointsPath = "/ceph/mstrobl/checkpoints"
modelsPath = "/ceph/mstrobl/models"
quantumPath = "/ceph/mstrobl/data_quantum"

batchSize = 16
epochs = 30

if __name__ == '__main__':

    x_train = np.load(f"{featurePath}/x_train_speech.npy")
    x_valid = np.load(f"{featurePath}/x_test_speech.npy")
    y_train = np.load(f"{featurePath}/y_train_speech.npy")
    y_valid = np.load(f"{featurePath}/y_test_speech.npy")


    q_train, q_valid = gen_quanv(x_train, x_valid, 2, output=quantumPath) 

