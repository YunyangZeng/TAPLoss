#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=debug \
  demucs.causal=1 \
  demucs.hidden=64 \
  demucs.resample=4 \
  batch_size=10 \
  revecho=0 \
  acoustic_loss=True \
  segment=10 \
  stride=2 \
  shift=16000 \
  shift_same=True \
  num_workers=16 \
  epochs=100 \
  ddp=1 $@
