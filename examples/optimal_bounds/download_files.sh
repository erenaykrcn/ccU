#!/bin/bash

remote_dir="/cluster/home/ekaracan/ccU/examples/optimal_bounds/cluster_scripts/results"

for layer in 13 14; do
  for time in 0.5 0.6 0.75 0.9; do
    fname="t${time}_layers${layer}.hdf5"
    echo "Downloading $fname..."
    sudo scp "ekaracan@euler.ethz.ch:${remote_dir}/${fname}" ./results/
  done
done



