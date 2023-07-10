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
num_processes=$NUM_PROCESSES
process_num=$PROCESS_NUM

startframe=$STARTFRAME
endframe=$ENDFRAME

# i=startframe+$((PROCESS_ID-1))
for (( i=$((startframe+($process_num-1))); i<endframe; i+=num_processes ))
do
    echo "(Process ${process_num}): frame #${i}/${endframe}" # Do not print the other frames

    if $run_occlusion
    then
        python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --process ${process_num} --occlusion --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} > /dev/null 
        python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_${process_num} --filename ${file} --frame ${i} --resolution ${resolution} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} --culling-r ${culling_r} > /dev/null 
        rm ./examples/vista_traces/lidar_${process_num}/lidar_3d*
    else
        python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --filename ${file} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} --process ${process_num} > /dev/null 
    fi
    # Use a pipe to output the progress somewhere?
    # For each process created, create a progress bar on the terminal to show the progress of each process
    # Use 'echo -ne'
done 

exit