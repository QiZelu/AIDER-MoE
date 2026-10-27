#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec /home/u202511081000084/miniconda3/envs/uve-copy/bin/python -B infer.py "$@"
