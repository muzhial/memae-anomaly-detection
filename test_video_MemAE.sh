#!/bin/bash
# testing MemAE on video dataset
python script_testing.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --Seed 1 \
    --ModelRoot ./models/ \
    --ModelFilePath results/model_out1/out1_epoch_0040.pt \
    --DataRoot /dataset/mz/outside_data/fault_vid \
    --OutRoot ./results/ \
    --Suffix Non
