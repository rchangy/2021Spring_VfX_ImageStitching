#!/bin/bash

python3 feature_detection.py ../data
python3 matching.py ../data
python3 stitching.py ../data