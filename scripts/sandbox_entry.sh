#!/bin/sh
set -eu

if [ "$#" -ne 13 ]; then
    echo "sandbox_entry.sh received an invalid argument count" >&2
    exit 64
fi

sandbox_temp=$1
venv_path=$2
model_path=$3
scripts_path=$4
output_path=$5
report_name=$6
artifact_name=$7
source_url=$8
expected_sha256=$9
cpu_seconds=${10}
memory_bytes=${11}
output_limit_bytes=${12}
sandbox_profile=${13}

worker_path="$scripts_path/architecture_worker.py"
landlock_path="$scripts_path/landlock_exec.py"
report_path="$output_path/$report_name"
python_executable=$(/usr/bin/readlink -f "$venv_path/bin/python")
python_runtime=$(/usr/bin/dirname "$(/usr/bin/dirname "$python_executable")")

/usr/bin/env -i \
    PATH=/usr/bin \
    HOME="$sandbox_temp" \
    TMPDIR="$sandbox_temp" \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    JSMI_SANDBOX_PROFILE="$sandbox_profile" \
    /usr/bin/prlimit \
        --cpu="$cpu_seconds" \
        --as="$memory_bytes" \
        --fsize="$output_limit_bytes" \
        --nofile=128 \
        --nproc=256 \
        -- \
    /usr/bin/setpriv \
        --no-new-privs \
        --bounding-set=-all \
        --inh-caps=-all \
        --ambient-caps=-all \
        -- \
    /usr/bin/python3 -I "$landlock_path" \
        --allow-read /usr \
        --allow-read /etc/ld.so.cache \
        --allow-read /dev/urandom \
        --allow-read /proc \
        --allow-read "$venv_path" \
        --allow-read "$python_runtime" \
        --allow-read "$scripts_path" \
        --allow-read "$model_path" \
        --allow-write /dev/null \
        --allow-write "$sandbox_temp" \
        --allow-write "$output_path" \
        -- \
    "$venv_path/bin/python" -I "$worker_path" \
        --model "$model_path" \
        --output "$report_path" \
        --artifact-name "$artifact_name" \
        --source-url "$source_url" \
        --expected-sha256 "$expected_sha256"
