#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

#experiments=("test1" "test2" "test3")
experiments=("fedavg_5clients" "fedavg_10clients" "fedavg_15clients" "fedavg_23clients" "fedavg_5clients_5e5" "fedavg_10clients_5e5" "fedavg_15clients_5e5" "fedavg_23clients_5e5")
echo "=============================================================================================="
echo "Starting Client Experiments"
echo "=============================================================================================="
for experiment in ${experiments[@]}; do
    echo "Starting experiment <$experiment> as nohup and directing output to output/$experiment.out"
    $base_path/../../venv/bin/python sync_base.py -c $experiment.yml
    echo "Experiment <$experiment> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing experiments"