import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
#Activate the cuda env
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"
print(os.environ.get("LD_LIBRARY_PATH"))
import glob
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from stqft.frontend import frontend, signal, transform
from stqft.stqft import stqft_framework
from stqft.stft import stft_framework

from qcnn.main_qsr import gen_train_from_wave, labels
from qcnn.models import attrnn_Model

datasetPath = "/ceph/mstrobl/dataset"
featurePath = "/ceph/mstrobl/features/"
checkpoints = "/ceph/mstrobl/checkpoints/"

batchSize = 16
epochs = 60

x_train = np.load(featurePath + "x_train_speech.npy")
x_valid = np.load(featurePath + "x_test_speech.npy")
y_train = np.load(featurePath + "y_train_speech.npy")
y_valid = np.load(featurePath + "y_test_speech.npy")


q_train = x_train
q_valid = x_valid

## For Quanv Exp.
early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                        verbose=1, patience=10, min_delta=0.0001)

metric = 'val_accuracy'

checkpoint = ModelCheckpoint('checkpoints/best_speech.hdf5', monitor=metric, 
                            verbose=1, save_best_only=True, mode='max')


model = attrnn_Model(q_train[0], labels)

model.summary()
# plot_model(model, to_file='model.png')

history = model.fit(
    x=q_train, 
    y=y_train,
    epochs=epochs, 
    callbacks=[checkpoint], 
    batch_size=batchSize, 
    validation_data=(q_valid,y_valid)
)

data_ix = time.strftime("%Y%m%d_%H%M")
#TODO: swtich to new model format
model.save(f"{checkpoints}/data_ix_speech.hdf5")