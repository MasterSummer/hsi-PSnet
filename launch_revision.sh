#!/usr/bin/env bash
# Run from an existing Git checkout. Working branch and files are preserved.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash launch_revision.sh --split-dir /path/to/split_4re [--rgb-root /path/to/rgb] [--zip /path/to/new.zip]'
  exit 2
fi
for arg in "$@"; do
  case "$arg" in --output|--output=*) echo 'The launcher chooses a fresh output directory; do not pass --output.' >&2; exit 2;; esac
done
repo_root=$(git rev-parse --show-toplevel)
revision_ref=${PSNET_REVISION_REF:-origin/codex/psnet-revision-package}
revision_sha=$(git rev-parse "${revision_ref}^{commit}")
python_bin=$(command -v "${PSNET_PYTHON:-python}")
jobs_root=${PSNET_JOBS_ROOT:-$repo_root/revision_jobs}
mkdir -p "$jobs_root"
job_dir=$(mktemp -d "$jobs_root/job_$(date +%Y%m%d_%H%M%S)_XXXXXX")
job_dir=$(cd "$job_dir" && pwd)
mkdir "$job_dir/code"
git -C "$repo_root" archive "$revision_sha" | tar -x -C "$job_dir/code"
printf '%s\n' "$revision_sha" > "$job_dir/CODE_COMMIT.txt"
printf '%s\n' "$python_bin" > "$job_dir/PYTHON.txt"
# Keep original cwd to resolve relative input arguments; the runner makes them absolute.
nohup bash -c '
  job_dir=$1; python_bin=$2; shift 2
  "$python_bin" -u "$job_dir/code/run_unattended_revision.py" "$@" --output "$job_dir/results"
  job_exit=$?
  printf "%s\n" "$job_exit" > "$job_dir/EXIT_CODE.txt"
  exit "$job_exit"
' psnet-job "$job_dir" "$python_bin" "$@" > "$job_dir/job.log" 2>&1 < /dev/null &
job_pid=$!
printf '%s\n' "$job_pid" > "$job_dir/PID.txt"
printf 'Launched PID %s\nJob directory: %s\nLog: %s/job.log\n' "$job_pid" "$job_dir" "$job_dir"
printf 'You may disconnect SSH. Later read results/STATUS.json and results/RESULTS.md; EXIT_CODE.txt is written when the process exits.\n'
