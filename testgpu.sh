#!/bin/bash

echo "Testing Theano with CUDA GPU, be sure to activate environment first"
THEANO_FLAGS=device=cuda0 python theanotest.py
