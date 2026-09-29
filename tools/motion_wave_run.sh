#!/bin/bash
# Stop + smile + wave clip chained to a walk clip, then seg + SAM-3D-Body fit.  tools/motion_wave_run.sh <subject> <walk clip> <id> <seed>
set -e
S=$1; WALK=$2; ID=$3; SEED=$4; W=work/$S; cd "$(dirname "$0")/.."
# b2crunner's tools (its docs/tools.md), in the environments they need; this host has no sam3dbody venv, so its
# SAM-3D-Body checkout stands in
RUN="python3 ${B2CRUNNER:-$HOME/Projects/b2crunner}/tools/run.py"
export B2CRUNNER_PYTHON_SAM3DBODY=${B2CRUNNER_PYTHON_SAM3DBODY:-$HOME/Projects/sam-3d-body/.venv/bin/python}
export B2CRUNNER_PATH_SAM3DBODY=${B2CRUNNER_PATH_SAM3DBODY:-$HOME/Projects/sam-3d-body}
A="She walks towards the camera, then slows down and comes to a stop, standing relaxed. She smiles warmly at the camera and gives a friendly wave hello: she lifts her right hand to shoulder height, palm open and facing the camera, and waves it side to side several times, then lowers her arm to her side and keeps smiling."
if [ ! -f $W/clips/$ID/wan/timing.json ]; then
  flock work/gpu.lock .venv/bin/python tools/motion_video.py $W $ID --action "$A" --az 25 --el 3 --radius 5.0 --target-dy -0.05 --seed $SEED --first $W/clips/$WALK/wan/0080.png
  # the kept first frame is the walk clip's last one; so is the identity reference
  flock work/gpu.lock $RUN wan_clip $W/clips/$ID > $W/clips/$ID/wan_log.txt 2>&1
fi
[ -d $W/clips/$ID/seg ] || flock work/gpu.lock $RUN seg_clip $W/clips/$ID > /dev/null 2>&1
[ -f $W/clips/$ID/video_fit.npz ] || flock work/gpu.lock $RUN video_fit $W/clips/$ID 2>&1 | grep "frames ->"
echo "motion_wave_run $S $ID done"
