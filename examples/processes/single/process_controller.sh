#!/bin/bash

### USER INPUT HERE ###
PROCESSES=5
LASFILE="(11)77502S_C1L1_02000_00000.las"
JSONFILE=velodyne_alpha_128.json # Do not put quotes here
observer_height=1.8
PAD_OUTPUTS=true

#TODO Parse the sensor config file from shell script and then pass the variables to each process
# You can input the sensor .json for now
RESOLUTION=0.11
PITCH_MIN=-25
PITCH_MAX=15
YAW_MIN=-180
YAW_MAX=180
RANGE=245 
CULLING_R=2

# Comment this out if you want to generate a trajectory or if you already have a pregenerated trajectory
python gen_traj.py --input examples/vista_traces/${LASFILE} --observer_height ${observer_height}

# Here you can run the vista output from a certain range if you need to
if $PAD_OUTPUTS
then
    # Run the entire road section, padded with the sensor range
    #STARTFRAME=$RANGE
    STARTFRAME=1560
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

export RESOLUTION
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

start=1
end=$PROCESSES

echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME}..."
echo "Input road section: ${LASFILE}"
echo "Input sensor configuration: ${JSONFILE}"
echo
echo "Starting ${PROCESSES} processes..."
start_time=`date +%s`


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
total_outputs=`ls "examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}" | wc -l`
expected_outputs=$(( $ENDFRAME - $STARTFRAME ))
if [[ $total_outputs -ne $expected_outputs ]]
then
    echo "Expected ${expected_outputs} outputs! (got ${total_outputs})"
    # exit 1
    # Find a way to generate the missing outputs
else
    echo "Opening MATLAB and generating graphs..."
    matlab -sd "~/Desktop/sensor-voxelization-cst/DataRate_fromCH" -r "data_rate_vista_automated('${SENSORPATH}', '${VISTAOUTPATH}', 1, 1)"
fi

#FOR DEBUGGING MATLAB FUNCTION, USING ALREADY GENERATED VISTA OUTPUTS
# NOTE THAT YOU MAY HAVE TO EDIT THE OUTPUT FOLDER
# data_rate_vista_automated("/home/mohamed/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/velodyne_alpha_128.json", "~/Desktop/vista/examples/vista_traces/lidar_output/03210N_C1R1_16000_20000_y_trimmed_resolution=0.11", true, true)