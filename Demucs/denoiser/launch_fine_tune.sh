#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dummy=debug_l2_waveform_loss\
  dset=dns \
  acoustic_loss=False \
  acoustic_loss_only=False \
  stft_loss=True \
  ac_loss_weight=0 \
  epochs=10 \
  ddp=1 $@
