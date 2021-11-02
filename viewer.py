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
import matplotlib as mpl

from cycler import cycler
import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")
from stqft.frontend import export, frontend
from qcnn.small_qsr import labels

def savePlot(name):
    plt.savefig(f"./{name}.pdf", format='pdf')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
    
frontend.setTheme(dark=False)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['axes.prop_cycle'] =   (cycler(color=[frontend.MAIN, frontend.HIGHLIGHT, frontend.MAIN, frontend.HIGHLIGHT])+
                                    cycler(linestyle=['-', '-', '--', '--']))

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

fri = frontend()

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

            nlabels = 10

            for it in range(10):
                row = floor(it/5)
                col = it - row*5
                oneHot = data[export.GENERICDATA]["y_train"][it*1000]
                y_idx = np.argmax(oneHot, axis=0)

                y_hat = data[export.GENERICDATA]["x_train"][it]
                y_hat_rs = np.reshape(y_hat,y_hat.shape[0:2])
                fri._show(yData=y_hat_rs, x1Data=None, sr = sr, title=f'STQFT_sim_n', ylabel="Frequency (Hz)", xlabel="Time (s)", plotType='librosa', xticks=[0, 1, 2, 3, 4])

            savePlot("trainFeatureWaveform")

        elif "quantumData" in filePath:
            print(f"Quantum Data:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from the first sample in the train set")

            fig, axs = plt.subplots(1,4, sharex=True, sharey=True)
            fig.set_size_inches(16,9)

            plt.tight_layout()


            q_train=data[export.GENERICDATA]['q_train']
            if q_train.shape[3]!=1:
                for i in range(4):
                    img = librosa.display.specshow(librosa.power_to_db(q_train[0,:,:,i], ref=np.min), ax=axs[i])

                    fig.colorbar(img, ax=axs[i], format='%+2.0f dB')
                    axs[i].set(title=f'Channel {i}')

                savePlot("trainQuantumData")

        elif "model" in filePath:
            print(f"Model:")
            print(f"{data[export.DESCRIPTION]}")
            print(f"Generating a plot from training history")

            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(10,6)
            plt.tight_layout()
            
            plt.plot(data[export.GENERICDATA]['history_loss'])
            plt.plot(data[export.GENERICDATA]['history_val_loss'])

            plt.plot(data[export.GENERICDATA]['history_acc'])
            plt.plot(data[export.GENERICDATA]['history_val_acc'])

            plt.title('Training / Validation History')
            plt.ylabel('Loss / Accuracy')
            plt.xlabel('Epochs')
            plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')

            savePlot("trainHistory_val_acc")

        else:
            print(f"not sure how to handle {filePath}")        


    except KeyError as e:
        print(f"Error while processing {filePath}: there was a keyerror: {e}")
    print()