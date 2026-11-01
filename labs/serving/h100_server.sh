#!/usr/bin/env bash
# Dry-run by default. Requires a pre-existing, digest-pinned image and local model.
set -euo pipefail
run=0
tp=1
multi=0
while (( $# )); do
  case "$1" in
    --run) run=1; shift;;
    --tp) tp="$2"; shift 2;;
    --allow-multi-gpu) multi=1; shift;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
: "${COURSE_VLLM_IMAGE:?set a locally available image@sha256:digest}"
: "${COURSE_MODEL_DIR:?set an absolute local model directory}"
: "${COURSE_GPU_IDS:?set an assigned GPU ID, or explicit comma-separated IDs}"
[[ "$COURSE_VLLM_IMAGE" == *@sha256:* ]] || { printf 'Pin image by digest.\n' >&2; exit 2; }
[[ "$COURSE_MODEL_DIR" == /* && -d "$COURSE_MODEL_DIR" ]] || { printf 'Model directory must exist and be absolute.\n' >&2; exit 2; }
[[ "$tp" =~ ^(1|2|8)$ ]] || { printf 'Supported lesson TP: 1, 2, 8.\n' >&2; exit 2; }
[[ "$COURSE_GPU_IDS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || { printf 'GPU IDs must be canonical nonnegative integers.\n' >&2; exit 2; }
IFS=, read -r -a gpu_ids <<< "$COURSE_GPU_IDS"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ ! -v "seen_gpu_ids[$gpu_id]" ]] || { printf 'GPU IDs must be distinct.\n' >&2; exit 2; }
  seen_gpu_ids[$gpu_id]=1
done
(( ${#gpu_ids[@]} == tp )) || { printf 'GPU count must equal TP.\n' >&2; exit 2; }
(( tp == 1 || multi == 1 )) || { printf 'Multiple GPUs require --allow-multi-gpu.\n' >&2; exit 2; }
name="llm-course-${UID}-$$"
# Docker requires extra quoting for a comma-separated --gpus device selector.
gpu_spec="\"device=${COURSE_GPU_IDS}\""
cmd=(docker run --rm --pull=never --name "$name" --gpus "$gpu_spec"
  --cpus 4 --memory 24g --shm-size 2g
  -p 127.0.0.1:18080:8000
  -v "$COURSE_MODEL_DIR:/model:ro"
  --entrypoint vllm "$COURSE_VLLM_IMAGE"
  serve /model --host 0.0.0.0 --port 8000 --tensor-parallel-size "$tp"
  --gpu-memory-utilization 0.35 --max-model-len 2048 --max-num-seqs 4
  --max-num-batched-tokens 2048 --enforce-eager)
printf 'Bounded 5-minute teaching server command:\n'
printf '%q ' "${cmd[@]}"; printf '\n'
if (( ! run )); then printf 'Dry run. Add --run after checking assigned GPU availability.\n'; exit 0; fi
docker image inspect "$COURSE_VLLM_IMAGE" >/dev/null
cleanup() { docker stop --time 10 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
timeout --signal=TERM --kill-after=20s 300s "${cmd[@]}"
