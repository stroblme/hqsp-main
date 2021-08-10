import sys
sys.path.append("./stqft")
sys.path.append("./qcnn")

import os
import glob
import shutil

from qcnn.main_qsr import labels

datasetPath = "/ceph/mstrobl/dataset"
testDatasetPath = "/ceph/mstrobl/testDataset"

portion = 10

for label in labels:
    datasetLabelFiles = glob.glob(f"{datasetPath}/{label}/*.wav")
    try:
        os.makedirs(f"{testDatasetPath}/{label}")
    except FileExistsError:
        pass

    all_wave = list()
    all_label = list()

    portDatsetLabelFiles = datasetLabelFiles[0::portion]
    print(f"Using {len(portDatsetLabelFiles)} out of {len(datasetLabelFiles)} files for label '{label}'")

    for datasetLabelFile in portDatsetLabelFiles:
        shutil.move(datasetLabelFile, datasetLabelFile.replace(datasetPath, testDatasetPath))