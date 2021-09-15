# from tkinter import *
# from tkinter.ttk import *
from math import ceil, floor
import pickle
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
import glob
import os
from qbstyles import mpl_style
import librosa
import librosa.display
import numpy as np
import random

import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")
from stqft.frontend import export, frontend
from qcnn.small_qsr import labels

def savePlot(name):
    plt.savefig(f"./{name}.png")

def melPlot(y_hat, sr=16000):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(y_hat, x_axis='time', y_axis='linear', sr=sr, fmax=sr/2, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')


def historyPlot(history, name):
    plt.plot(history['history_loss'])
    plt.plot(history['history_val_loss'])

    plt.plot(history['history_acc'])
    plt.plot(history['history_val_acc'])

    plt.title('model loss')
    plt.ylabel('loss/acc')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper left')

    fig = plt.gcf()
    fig.set_size_inches(16,9)
    
    # plt.show()
    plt.savefig(f"./{name}_val_acc.png")

    
frontend.setTheme(dark=True)

cdir = "/storage/mstrobl/versioning/"
ignoreList = ["venv", ".vscode", ".git"]

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
            print(f"Generating some plots from the random sample in the train set")

            fig, axs = plt.subplots(2,5, sharex=True, sharey=True)
            fig.set_size_inches(24,10)

            plt.tight_layout

            sr = 16000

            for it in range(10):
                row = floor(it/5)
                col = it - row*5
                oneHot = data[export.GENERICDATA]["y_train"][random.randint(0,len(data[export.GENERICDATA]["y_train"])-1)]
                y_idx = np.argmax(oneHot, axis=0)

                y_hat = data[export.GENERICDATA]["x_train"][it]
                y_hat_rs = np.reshape(y_hat,y_hat.shape[0:2])
                img = librosa.display.specshow(y_hat_rs, x_axis='time', y_axis='linear', sr=sr, fmax=sr/2, ax=axs[row][col])

                fig.colorbar(img, ax=axs[row][col], format='%+2.0f dB')
                axs[row][col].set(title=f'"{labels[y_idx]}"')
                # print(f"{row}+{col}")

            savePlot("trainFeatureWaveform")

        # elif "quantumData" in filePath:
        #     print(f"Quantum Data:")
        #     print(f"{data[export.DESCRIPTION]}")
        #     print(f"Generating a plot from the first sample in the train set")
        #     melPlot(data[export.GENERICDATA]["q_train"][0], "trainFeatureQuantum")
        elif "model" in filePath:
            print(f"Model:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from training history")
            historyPlot(data[export.GENERICDATA], "trainHistory")
        # elif "errors" in filePath:
        #     print(f"Model:")
        #     print(f"{data[export.DESCRIPTION]}")
        #     print(f"Generating a plot from training history")
        #     historyPlot(data[export.GENERICDATA]["history"], "trainHistory")
        else:
            print(f"not sure how to handle {filePath}")        


    except KeyError as e:
        print(f"Error while processing {filePath}: there was a keyerror: {e}")
    print()