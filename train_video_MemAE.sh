python script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 2 \
    --Seed 1 \
    --SaveCheckInterval 20 \
    --TextLogInterval 50 \
    --IsTbLog True \
    --IsDeter True \
    --DataRoot /dataset/mz/outside_data/fault_vid \
    --ModelRoot ./results/ \
    --Suffix Non
