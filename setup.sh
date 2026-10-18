#!/usr/bin/env bash
# Install Python deps and fetch the Franka Panda model (MuJoCo Menagerie, Apache-2.0).
set -e
pip install -r requirements.txt
if [ ! -d menag/franka_emika_panda ]; then
  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie.git menag
  (cd menag && git sparse-checkout set franka_emika_panda)
fi
echo "Done. Headless Linux: export MUJOCO_GL=osmesa (apt install libosmesa6) or MUJOCO_GL=egl"
