#!/bin/bash

EPOCH_IDX=105

python -m pytorch_fid ../dataset/zalando-hd-resized/test/image ../v2/outputs/epoch-${EPOCH_IDX}/unpaired/cfg_5