#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

experiments=("fedavg_r100_e1_lr0000001" "fedavg_r100_e1_lr000001" "fedavg_r100_e1_lr00001")
for experiment in ${experiments}; do
    if [ ! -f output/$experiment.txt ]; then
        touch $base_path/experiment_output/$experiment.txt
    fi
    $base_path/../../venv/bin/python sync_base.py -c $experiment.yml > experiment_output/$experiment.txt
done