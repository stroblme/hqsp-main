# Hybrid Quantum Speech Processing - Main repository

This is the root directory of the *hqsp* project

## Usage

### Training

The training can be executed by running [train.py](train.py).
This includes:

- check versioning for changes
- generating waveforms (spectrograms)
- generating quantum data
- training the network

Waveforms can be loaded from disk by including <code>--waveform=0</code> as argument.\
Quantum data can be loaded from disk by including <code>--quantum=0</code> as argument.\
The Pixel-Channel-Mapping can be activated by setting <code>--quantum=-1</code>.\
Checking the versioning directory can be disabled by <code>--checkTree=-1</code>.
Actual training can be disabled by <code>--train=0</code>.

Paths in the script need to be adapted to your needs.

### Testing

Similar to training:

- check versioning for changes
- generating waveforms (spectrograms)
- generating quantum data
- testing the network

Waveforms can be loaded from disk by including <code>--waveform=0</code> as argument.\
Quantum data can be loaded from disk by including <code>--quantum=0</code> as argument.\
The Pixel-Channel-Mapping can be activated by setting <code>--quantum=-1</code>.\
Checking the versioning directory can be disabled by <code>--checkTree=-1</code>.

Paths in the script need to be adapted to your needs.

## Structure

General training procedure: [train.py](train.py)
General testing procedure: [test.py](test.py)
Quick evaluations and testing: [eval.py](eval.py)
Extraction of test data: [extractTestData.py](extractTestData.py)
Training, loading and evaluation of model: [fitModel.py](fitModel.py)
Spectrogram generation, global parameter storage and multiprocessing: [generateFeatures.py](generateFeatures.py)
Experiment viewer: [viewer.py](viewer.py)
