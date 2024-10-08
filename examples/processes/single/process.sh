#!/bin/bash

for i in {0..571} 
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/74202W_C1L1_L1L1_08000_06000.las --frame ${i} --range 245
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar --frame ${i} --resolution 0.1 --yaw-min -180 --yaw-max -180 --pitch-min -21 --pitch-max 19
    rm ./examples/vista_traces/lidar/lidar_3d*
done