#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dummy=waveform+1_pho_seg_ac_loss\
  dset=dns \
  acoustic_loss=True \
  acoustic_loss_only=False \
  stft_loss=False \
  ac_loss_weight=1 \
  stft_loss_weight=0.0 \
  epochs=40 \
  ddp=1 $@
