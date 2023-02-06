#!/usr/bin/env bash

set -e

print_sep="=============================="

eval_command() {
    local title="$1"
    local command="$2"

    echo "$print_sep $title $print_sep"
    eval "$command"
    echo
}

eval_command CONDA "conda info 2>/dev/null || echo \"Conda not found\""
eval_command PYTHON "which python && python -V"
eval_command PIP "python -m pip -V"
eval_command PYLINT "python -m pylint --version"
eval_command PYTEST "python -m pytest --version"
eval_command BLACK "python -m black --version"
eval_command ISORT "python -m isort --version"
eval_command PRE-COMMIT "python -m pre_commit --version"
eval_command REaLTabFormer "python -m realtabformer --version 2>/dev/null || echo \"REaLTabFormer not found\""
