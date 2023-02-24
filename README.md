# TAPLoss

Official Pytorch Implementation of [TAPLoss: A Temporal Acoustic Parameter Loss For Speech Enhancement](https://arxiv.org/abs/2302.08088) accepted to ICASSP 2023.

## Prerequisites

TAPLoss requires **Numpy** and **Pytorch**, install versions that are compatible with your project. \
Refer to [Demucs](Demucs/denoiser/README.md) and [FullSubNet](FullSubNet/README.md) for installing required packages for **Demucs** and **FullSubNet**.

## Dataset

Download the DNS Interspeech 2020 dataset from [here](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master) and follow the instructions to prepare the dataset.

## Use TAPLoss in your own project

1. You'll need the  `TAPLoss.py` and `TAP_estimator.py` located under `TAPLoss/`, you'll also find a `.pt` file under the same directory, which is the TAP estimator model checkpoint.
```python
from TAPLoss import AcousticLoss
```
2. Initialize TAPLoss 
```python
TAPloss = AcousticLoss(loss_type, acoustic_model_path,paap , paap_weight_path)
# loss_type: choose one from ["l2", "l1", "frame_energy_weighted_l2", "frame_energy_weighted_l1"]
# acoustic_model_path: path to the .pt TAP estimator model checkpoint.
# paap: set to True if you want to use paaploss, default is False.
# paap_weight_path: path to the paap weights .npy file, must specify if paap == True .
```
3. Call TAPloss as a function
```python
loss = TAPloss(clean_waveform, enhan_waveform, mode)
# clean_waveform: waveform of clean audio, sampled at 16khz.
# enhan_waveform: waveform of enhanced audio, sampled at 16khz.
# mode: "train" if your enhancement model is in train mode,
#        "eval"  if your enhancement model is in eval mode,
#        gradients won't be calculated in eval mode.
```

## Related Resources

More details about the official implementation of [PAAPLoss: A Phonetic-Aligned Acoustic Parameter Loss for Speech Enhancement](https://arxiv.org/abs/2302.08095) can be found at https://github.com/muqiaoy/PAAP. 
