#! /bin/bash

source .env

# delete everything that is saved
rm -rf *.pkl
rm -rf combined_images/
rm -rf plt_images/
rm -rf frame_images/
rm -rf fov/

# replay
python3 full_replay.py