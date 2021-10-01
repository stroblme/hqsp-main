import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"
print(os.environ.get("LD_LIBRARY_PATH"))
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from qcnn.small_qsr import labels
from qcnn.models import attrnn_Model, custom_attrnn_Model

datasetPath = "/ceph/mstrobl/dataset"
featurePath = "/ceph/mstrobl/features"
modelsPath = "/ceph/mstrobl/models"
quantumPath = "/ceph/mstrobl/data_quantum"
checkpointsPath = "/ceph/mstrobl/checkpoints"


def fit_model(q_train, y_train, q_valid, y_valid, cpPath, epochs, batchSize, ablation=False):
    ## For Quanv Exp.
    early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                            verbose=1, patience=10, min_delta=0.0001)

    metric = 'val_accuracy'

    checkpoint = ModelCheckpoint(cpPath, monitor=metric, 
                                verbose=1, save_best_only=True, mode='max')


    model = custom_attrnn_Model(q_train[0], labels, ablation=ablation)
    # model = attrnn_Model(q_train[0], labels, ablation=ablation)

    model.summary()

    history = model.fit(
        x=q_train, 
        y=y_train,
        epochs=epochs, 
        callbacks=[checkpoint], 
        batch_size=batchSize, 
        validation_data=(q_valid,y_valid)
    )
    return model, history

# if __name__ == '__main__':

#     x_train = np.load(f"{featurePath}/x_train_speech.npy")
#     x_valid = np.load(f"{featurePath}/x_test_speech.npy")
#     y_train = np.load(f"{featurePath}/y_train_speech.npy")
#     y_valid = np.load(f"{featurePath}/y_test_speech.npy")


#     q_train = np.load(f"{quantumPath}/quanv_train.npy")
#     q_valid = np.load(f"{quantumPath}/quanv_test.npy")

#     model = fit_model(q_train, y_train, q_valid, y_valid, checkpointsPath)

#     data_ix = time.strftime("%Y%m%d_%H%M")
#     model.save(f"{modelsPath}/model_{time.time()}")