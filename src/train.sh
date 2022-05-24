#!/bin/bash

# The first phase: No local, no event

python main.py   --multiprocessing-distributed --world-size 1   --rank 0  --smooth-labelling  \
      --saved_fn        'leo_1st_phase' \
    --no-val          \
    --batch_size      8 \
    --num_workers     4 \
      --lr              0.001 \
      --lr_type         'step_lr' \
      --lr_step_size    10 \
      --lr_factor       0.1 \
        --global_weight   5. \
        --seg_weight      1. \
        --no_local        \
        --no_event        \
      --smooth-labelling

# The second phase: Freeze the segmentation and the global modules

python main.py   --multiprocessing-distributed --world-size 1   --rank 0  --smooth-labelling   \
    --saved_fn        'leo_2nd_phase'                                         \
    --no-val                                                                  \
    --batch_size      8                                                       \
    --num_workers     4                                                       \
        --lr              0.001                                               \
        --lr_type         'step_lr'                                           \
        --lr_step_size    10                                                  \
        --lr_factor       0.1                                                 \
    --global_weight   0.                                                  \
    --seg_weight      0.                                                  \
    --event_weight    2.                                                  \
    --local_weight    1.                                                  \
        --pretrained_path ../checkpoints/leo_1st_phase/leo_1st_phase_epoch_30.pth \
    --overwrite_global_2_local                                            \
    --freeze_seg                                                          \
    --freeze_global                                                       \

# The third phase: Finetune all modules

python main.py   --multiprocessing-distributed --world-size 1   --rank 0  --smooth-labelling  \
    --saved_fn 'leo_3rd_phase' \
    --no-val \
    --batch_size 8 \
    --num_workers 4 \
        --lr 0.0001 \
        --lr_type 'step_lr' \
        --lr_step_size 10 \
        --lr_factor 0.2 \
    --global_weight 1. \
    --seg_weight 1. \
    --event_weight 1. \
    --local_weight 1. \
        --pretrained_path ../checkpoints/leo_2nd_phase/leo_2nd_phase_epoch_30.pth \
