# from tkinter import *
# from tkinter.ttk import *
import pickle
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
import glob
import os
from qbstyles import mpl_style
import librosa
import librosa.display
import numpy as np

from stqft.frontend import export, frontend

def melPlot(y_hat, name, sr=16000):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(y_hat, x_axis='time', y_axis='linear', sr=sr, fmax=sr/2, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    # plt.show()
    plt.savefig(f"./{name}.png")

def historyPlot(history, name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plt.show()
    plt.savefig(f"./{name}.png")
frontend.setTheme(dark=True)

cdir = "/ceph/mstrobl/versioning/"
ignoreList = ["venv", ".vscode"]

content = os.listdir(cdir)
folderList = list()

for c in content:
    if os.path.isdir(cdir+c):
        if c not in ignoreList:
            folderList.append(c)

print(f"Found {len(folderList)} folders in current directory:\n {folderList}")


selection = ""
if len(folderList) == 1:
    selection = folderList[0]
else:
    while(selection not in folderList):
        idx = input("Choose the desired datafolder as index (starting from 1)\n")
        try:
            selection = folderList[int(idx)-1]
        except IndexError:
            continue

    print(f"Showing {selection} ...")

fileList = glob.glob(f"{cdir + selection}/*.p")
pt = 0

for filePath in fileList:
    try:
        data = pickle.load(open(filePath,'rb'))
    except Exception as e:
        print(f"Error loading {filePath}: {e}")
        continue

    try:
        if "dataset" in filePath:
            print(f"Description of dataset:")
            print(f"{data[export.DESCRIPTION]}")
        elif "waveformData" in filePath:
            print(f"Waveforms:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from the first sample in the train set")
            y_hat = data[export.GENERICDATA]["x_train"][0]
            y_hat_s = np.reshape(y_hat,y_hat.shape[0:2])
            melPlot(y_hat_s, "trainFeatureWaveform")
        # elif "quantumData" in filePath:
        #     print(f"Quantum Data:")
        #     print(f"{data[export.DESCRIPTION]}")
        #     print(f"Generating a plot from the first sample in the train set")
        #     melPlot(data[export.GENERICDATA]["q_train"][0], "trainFeatureQuantum")
        elif "model" in filePath:
            print(f"Model:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from training history")
            historyPlot(data[export.GENERICDATA]["history"], "trainHistory")
        elif "errors" in filePath:
            print(f"Model:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from training history")
            historyPlot(data[export.GENERICDATA]["history"], "trainHistory")
        else:
            print(f"not sure how to handle {filePath}")        

    except KeyError as e:
        print(f"Error while processing {filePath}: there was a keyerror: {e}")