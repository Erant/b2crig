#!/bin/bash
# Generate the two chained motion clips (walk; stop + smile + wave) for a subject and fit them with SAM-3D-Body.
#   tools/motion_video_run.sh <subject> [seed]
set -e
S=$1; SEED=${2:-0}; W=work/$S; cd "$(dirname "$0")/.."
# b2crunner's tools (its docs/tools.md), in the environments they need; this host has no sam3dbody venv, so its
# SAM-3D-Body checkout stands in
RUN="python3 ${B2CRUNNER:-$HOME/Projects/b2crunner}/tools/run.py"
export B2CRUNNER_PYTHON_SAM3DBODY=${B2CRUNNER_PYTHON_SAM3DBODY:-$HOME/Projects/sam-3d-body/.venv/bin/python}
export B2CRUNNER_PATH_SAM3DBODY=${B2CRUNNER_PATH_SAM3DBODY:-$HOME/Projects/sam-3d-body}
A1="She immediately lowers her arms from the A pose to her sides and starts walking forward towards the camera, taking several steps with a relaxed, natural walking gait: legs stepping, arms swinging gently, hips and shoulders moving with each step. She keeps walking and gets closer to the camera."
A2="She walks towards the camera, then slows down and comes to a stop, standing relaxed. She smiles warmly at the camera and gives a friendly wave hello: she lifts her right hand to shoulder height, palm open and facing the camera, and waves it side to side several times, then lowers her arm to her side and keeps smiling."
run_clip () {  # id action [first]
  local id=$1 act=$2 first=$3
  if [ ! -f $W/clips/$id/wan/timing.json ]; then
    flock work/gpu.lock .venv/bin/python tools/motion_video.py $W $id --action "$act" --az 25 --el 3 --radius 5.0 --target-dy -0.05 --seed $SEED ${first:+--first $first}
    flock work/gpu.lock $RUN wan_clip $W/clips/$id > $W/clips/$id/wan_log.txt 2>&1
  fi
  [ -d $W/clips/$id/seg ] || flock work/gpu.lock $RUN seg_clip $W/clips/$id > /dev/null 2>&1
  [ -f $W/clips/$id/video_fit.npz ] || flock work/gpu.lock $RUN video_fit $W/clips/$id 2>&1 | grep video_fit
}
run_clip mv_walk_s$SEED "$A1"
run_clip mv_wave_s$SEED "$A2" $W/clips/mv_walk_s$SEED/wan/0080.png
echo "motion_video_run $S done"
