# File/program structure information

[Back to README.md](README.md#occlusion)

This document contains an overview of which programs/functions within these programs are called, in order for certain tasks. I wrote these notes as reference down in case if you want to make changes to the current code.

## Creating Vista outputs (running the process controller)

### ``convert_single.py``

First of all, our .las point cloud and sensor configuration is defined in ``examples/processes/single/process_controller.sh``. Regardless of the version of process controller that you choose for generating the outputs, it will run each process in the background. (``process_{1_TO_6}.sh`` for ``process_controller.sh`` and ``process_controller_python.sh``, for ``process_controller_v2.sh``, the processes spawned are from ``process_i.sh``.)

<details>
  <summary>Here is the code of the first process.</summary>

```shell
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
```
</details>
<br>

For each process, we will perform ``convert_single.py`` (perform a rigid body translation, cull far away points and rotation at point $i$).

- In this case, the ``main()`` function is called under the main guard, and arguments from the shell script are passed into the script itself.
- After performing our computations, we save our intermediate point cloud as an temporary .h5 file which we will pass into ``sim_lidar.py``.

### ``sim_lidar.py``

Here is where we actually perform the viewpoint simulation. For the most part, we are using the VISTA simulator to simulate occlusion; at least for our application of simulating and calculating the data rate, ``vista/entities/sensors/lidar_utils/LidarSynthesis.py`` is the only function that is relevant.

That aside, here is the full path of what programs and functions are called when we run the python script.

Code snippet of ``sim_lidar.py`` (I will explain the functions a bit later on):

```python
import argparse
import numpy as np
import os
import cv2

import vista


def main(args):
    world = vista.World(
        args.trace_path, trace_config={"road_width": 4, "master_sensor": "lidar_3d"}
    )
    car = world.spawn_agent(
        config={
            "length": 5.0,
            "width": 2.0,
            "wheel_base": 2.78,
            "steering_ratio": 14.7,
            "lookahead_road": True,
        }
    )
    lidar_config = {
        "yaw_fov": (args.yaw_min, args.yaw_max),
        "pitch_fov": (args.pitch_min, args.pitch_max),
        "frame": args.frame,
        "yaw_res": args.resolution,
        "pitch_res": args.resolution,
        "downsample": args.downsample,
        "culling_r": args.culling_r,
        "roadsection_filename": args.filename
    }
    lidar = car.spawn_lidar(lidar_config)
    display = vista.Display(world)

    world.reset()
    display.reset()

    # while not car.done:
    action = follow_human_trajectory(car)

    ## Removed because we are not really using this
    # vis_img = display.render()
    # cv2.imshow("Visualize LiDAR", vis_img[:, :, ::-1])
    # cv2.waitKey(10000)
```

First of all