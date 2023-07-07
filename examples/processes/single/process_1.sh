#!/bin/bash

file=$LASFILE

resolution=$RESOLUTION
pitch_min=$PITCH_MIN
pitch_max=$PITCH_MAX
yaw_min=$YAW_MIN
yaw_max=$YAW_MAX
range=$RANGE
culling_r=$CULLING_R
run_occlusion=$RUN_OCCLUSION
num_processes=$PROCESSES

startframe=$STARTFRAME
endframe=$ENDFRAME

for (( i=startframe+0; i<endframe; i+=num_processes ))
do
    if $run_occlusion
    then
        python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --process 1 --occlusion --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} 
        python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_1 --filename ${file} --frame ${i} --resolution ${resolution} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} --culling-r ${culling_r}
        rm ./examples/vista_traces/lidar_1/lidar_3d*
    else
        python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --filename ${file} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} --process 1
    fi
done 

exit