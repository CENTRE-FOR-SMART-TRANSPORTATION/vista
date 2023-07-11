#!/bin/bash

### USER INPUT BEGINS HERE ###

NUM_PROCESSES=6
LASFILE="(13)-noveg-84508S_C1L1_28000_26000.las"
JSONFILE=velodyne_alpha_128.json # Do not put quotes here
observer_height=1.8
PAD_OUTPUTS=true # Only generate outputs that have the entire road in range (This saves some time)
GRAPHICAL_VIA_PYTHON=false # MATLAB otherwise

# TODO Parse the sensor config file from shell script and then pass the variables to each process
# You can input the sensor .json for now.
## Angular resolutions
#RESOLUTION_AZI=0.11 # Will implement later...
#RESOLUTION_ELE=0.11
RESOLUTION=0.11

## Sensor parameters
PITCH_MIN=-25
PITCH_MAX=15
YAW_MIN=-180
YAW_MAX=180
RANGE=245 
CULLING_R=2

# Generate our trajectory, should it not exist and is not valid (there must be 5 csvs)
num_csvs=`find "./examples/Trajectory/${LASFILE%.*}/" -type f -name "*.csv" | wc -l`

if [[ -d "./examples/Trajectory/${LASFILE%.*}" ]] && [[ $num_csvs -eq 5 ]]; then
  echo "Trajectory exists already, skipping generation."
else
  echo "Generating trajectory."
  python gen_traj.py --input examples/vista_traces/${LASFILE} --observer_height ${observer_height}
fi

### USER INPUT BEGINS HERE ###

# Define the starting and ending frames
if $PAD_OUTPUTS
then
    # Run the entire road section, padded with the sensor range
    STARTFRAME=$RANGE
    ENDFRAME=$((`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`-$RANGE))

else
    STARTFRAME=0
    ENDFRAME=`cat examples/Trajectory/${LASFILE%.*}/forwards.csv | wc -l`
fi

# Uncomment these if want to generate your outputs in some specific range
#STARTFRAME=SOMEFRAME
#ENDFRAME=SOMEOTHERFRAME

### USER INPUT ENDS HERE ###

# Just check for valid input
if (( $STARTFRAME > $ENDFRAME ))
then
    echo "Cannot have starting frame greater than end frame! (${STARTFRAME} > ${ENDFRAME})"
    exit 1
fi

# Export our variables for our subprocesses
export RESOLUTION
export PITCH_MIN
export PITCH_MAX
export YAW_MIN
export YAW_MAX
export RANGE
export CULLING_R

export LASFILE
export STARTFRAME
export ENDFRAME
export PAD_OUTPUTS
export RUN_OCCLUSION
export NUM_PROCESSES


echo
<<<<<<< Updated upstream
echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME} with ${NUM_PROCESSES} processes!"
=======
echo "Computing output from road point ${STARTFRAME} to road point ${ENDFRAME} with ${NUM_PROCESSES} processes."
>>>>>>> Stashed changes
echo "Input road section: ${LASFILE}"
echo "Input sensor configuration: ${JSONFILE}"
if $GRAPHICAL_VIA_PYTHON
then
  # https://askubuntu.com/questions/528928/how-to-do-underline-bold-italic-strikethrough-color-background-and-size-i
  echo -e '\e[48;2;240;143;104mGraphical outputs: Python\e[49m'
else
  echo -e '\e[48;2;240;143;104mGraphical outputs: MATLAB\e[49m'
fi

echo
start_time=`date +%s` # Just for timing purposes

echo -e "\e[3m\e[1mType \"sh kill_processes.sh\" to exit the processes\e[0m"

# This method will have background processes running unless if you explicitly kill them
for (( PROCESS_NUM=1; PROCESS_NUM<=NUM_PROCESSES; PROCESS_NUM++ ))
do
    export PROCESS_NUM
    bash ./examples/processes/single/process_i.sh &
done
wait # Wait for eveyrthing to finish.

end_time=`date +%s` # Just for timing purposes

echo "All processes have finished."
echo
echo -e "\e[3mVista simulation took $((${end_time}-${start_time})) seconds.\e[23m"


# Obtain the necessary paths for calling MATLAB/Python (this really should be reworked)
# This is hard coded, and I really should be using some other method to get the paths
SENSORPATH=~/Desktop/sensor-voxelization-cst/DataRate_fromCH/sensors/${JSONFILE}
VISTAOUTPATH="~/Desktop/vista/examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}"


# Check for total outputs; sanity check
total_outputs=`ls "examples/vista_traces/lidar_output/${LASFILE%.*}_resolution=${RESOLUTION}" | wc -l`
expected_outputs=$(( $ENDFRAME - $STARTFRAME ))

if [[ $total_outputs -ne $expected_outputs ]]
then
    echo -e "\e[31mExpected ${expected_outputs} outputs! (got ${total_outputs})\e[39m"
    # Find a way to generate the missing outputs or something later on?
    # I could create a pipe from check_missing and then start a process from that...
    # For now I will just print the missing files.
    python check_missing.py --output ${VISTAOUTPATH} --startframe ${STARTFRAME} --endframe ${ENDFRAME}
else
    echo "Generating graphs..."
    if $GRAPHICAL_VIA_PYTHON
    then
        python data_rate_vista.py --config ${SENSORPATH} --scenes ${OUTPATH} --numScenes 1
    else
        # I really need to update the MATLAB code to not use the last two options
        # Syntax:
        # data_rate_vista_automated(SENSOR_PATH, VISTA_OUTPATH, ENABLE_GRAPHICAL, PADDED)
        matlab -sd "~/Desktop/sensor-voxelization-cst/DataRate_fromCH" -r "data_rate_vista_automated('${SENSORPATH}', '${VISTAOUTPATH}', 1, 1)"
    fi
fi