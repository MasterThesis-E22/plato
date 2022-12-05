#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

declare -a experiments=(
"low-delay_low-interval"
"high-delay_low-interval"
"low-delay_high-interval"
"high-delay_high-interval"
"low-hinge"
"high-hinge"
"low-poly"
"high-poly"
"multiple_epochs"
            )

echo "=============================================================================================="
echo "Starting Benchmark Async experiments"
echo "=============================================================================================="
for experiment in ${experiments[@]}; do
    echo "Starting experiment <$experiment>"
    $base_path/../../venv/bin/python async_base.py -c cifar10/$experiment.yml
    echo "Experiment <$experiment> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing experiments"