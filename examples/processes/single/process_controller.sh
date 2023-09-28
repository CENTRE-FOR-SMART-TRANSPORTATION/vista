#!/bin/bash

### USER INPUT HERE ###
PROCESSES=6
LASFILE="(16)-noveg-81604S_C1L1_04000_00000.las"
JSONFILE=velodyne_alpha_128.json # Do not put quotes here
observer_height=1.8
PAD_OUTPUTS=true

#TODO Parse the sensor config file from shell script and then pass the variables to each process
# You can input the sensor .json for now
RESOLUTION_hz=0.11
RESOLUTION_vl=0.11
PITCH_MIN=-25
PITCH_MAX=15
YAW_MIN=-180
YAW_MAX=180
RANGE=245 
CULLING_R=2

## Define the range that we will write voxelized scenes using MATLAB, if we are writing them
write_voxel=false
VOXEL_OUT_START=500
VOXEL_OUT_END=510

# Comment this out if you want to generate a trajectory or if you already have a pregenerated trajectory
python gen_traj.py --input examples/vista_traces/${LASFILE} --observer_height ${observer_height}

# Here you can run the vista output from a certain range if you need to
if $PAD_OUTPUTS
then
    # Run the entire road section, padded with the sensor range
    STARTFRAME=$RANGE
    ENDFRAME=$((`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`-$RANGE))
else
    STARTFRAME=0
    ENDFRAME=`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`
    # ENDFRAME=6 # Uncomment this if you want to run in a certain range
fi

### USER INPUT ENDS HERE ###

if (( $STARTFRAME > $ENDFRAME ))
then
    echo "Cannot have starting frame greater than end frame! (${STARTFRAME} > ${ENDFRAME})"
    exit 1
fi

# Export our variables for our subprocesses
export RESOLUTION_hz
export RESOLUTION_vl
export PITCH_MIN
export PITCH_MAX
export YAW_MIN
export YAW_MAX
export RANGE
export CULLING_R
export PROCESSES

export LASFILE
export STARTFRAME
export ENDFRAME
export PAD_OUTPUTS


## Display information for the user
echo
echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME}..."
echo "Input road section: ${LASFILE}"
echo "Input sensor configuration: ${JSONFILE}"
if $write_voxel
then
    echo "Outputs from ${VOXEL_OUT_START} to ${VOXEL_OUT_END} will be written!"
fi
echo
echo "Starting ${PROCESSES} processes..."
start_time=`date +%s`


start=1
end=$PROCESSES
for (( i=$start; i<=$PROCESSES; i++ ))
do
    #xterm -T "Process ${i}" -hold -e bash examples/processes/single/process_${i}.sh &
    xterm -T "Process ${i}" -e bash examples/processes/single/process_${i}.sh &

    echo "Process ${i} of ${end} started"
done

wait # Waits until every single process has finished

echo "All processes have finished."
echo


SENSORPATH=~/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/${JSONFILE}
VISTAOUTPATH="~/Desktop/vista/examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}"

end_time=`date +%s`
echo "Vista simulation took $((${end_time}-${start_time})) seconds."

# Check if all Vista scenes within our range were generated
total_outputs=`ls "examples/vista_traces/lidar_output/"${LASFILE%.*}"_resolution=${RESOLUTION}/" | wc -l` # This is incredibly finnicky
expected_outputs=$(( $ENDFRAME - $STARTFRAME ))
if [[ $total_outputs -ne $expected_outputs ]]
then
    echo "Expected ${expected_outputs} outputs! (got ${total_outputs})"
    # exit 1
    # Find a way to generate the missing outputs
else
    echo "Opening MATLAB and generating graphs..."
    # Syntax:
    # data_rate_vista_automated(SENSOR_PATH, 
    #                           VISTA_OUTPATH, 
    #                           ENABLE_GRAPHICAL, 
    #                           PADDED, 
    #                           VOXELOUT_START, 
    #                           VOXELOUT_END, 
    #                           WRITE_VOXEL)
    if $write_voxel
    then
        matlab -sd "~/Desktop/sensor-voxelization-cst/DataRate_fromCH" -r "data_rate_vista_automated('${SENSORPATH}', '${VISTAOUTPATH}', 1, 1, ${VOXEL_OUT_START}, ${VOXEL_OUT_END}, 1)"
    else
        matlab -sd "~/Desktop/sensor-voxelization-cst/DataRate_fromCH" -r "data_rate_vista_automated('${SENSORPATH}', '${VISTAOUTPATH}', 1, 1, ${VOXEL_OUT_START}, ${VOXEL_OUT_END}, 0)"
    fi
fi

#FOR DEBUGGING MATLAB FUNCTION, USING ALREADY GENERATED VISTA OUTPUTS
# NOTE THAT YOU MAY HAVE TO EDIT THE OUTPUT FOLDER
# data_rate_vista_automated("/home/mohamed/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/velodyne_alpha_128.json", "~/Desktop/vista/examples/vista_traces/lidar_output/(7)-newcut-01104W_C1L1_16000_12000_resolution=0.11", true, true, 500, 510, false)