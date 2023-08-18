# File/program structure information

[Back to README.md](README.md#occlusion)

This document contains an overview of which programs/functions within these programs are called, in order for certain tasks. I wrote these notes as reference down in case if you want to make changes to the current code.

## Contents

1. [Creating Vista outputs](#creating-vista-outputs-running-the-process-controller)
2. [Files/tools within the repository](#files-and-tools)

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

<details>
    <summary>Here is a code snippet of sim_lidar.py. I will explain the details later on, but I commented the code snippet a bit.</summary>

```python
import argparse
import numpy as np
import os
import cv2

import vista


def main(args):

    # This specifies the world where all agents and sensors will live in
    # Agents and sensors will be attached to this class
    world = vista.World( # source: vista/core/World.py
        args.trace_path, trace_config={"road_width": 4, "master_sensor": "lidar_3d"}
    )

    # I am pretty sure that the car configuration isn't 
    # used for our data rate calculations.
    car = world.spawn_agent( # source: vista/entities/agents/Car.py
        config={
            "length": 5.0,
            "width": 2.0,
            "wheel_base": 2.78,
            "steering_ratio": 14.7,
            "lookahead_road": True,
        }
    )

    lidar_config = { # The configuration that we have passed
        "yaw_fov": (args.yaw_min, args.yaw_max),
        "pitch_fov": (args.pitch_min, args.pitch_max),
        "frame": args.frame,
        "yaw_res": args.resolution,
        "pitch_res": args.resolution,
        "downsample": args.downsample,
        "culling_r": args.culling_r,
        "roadsection_filename": args.filename
    }

    lidar = car.spawn_lidar(lidar_config) # src: vista/entities/sensors/Lidar.py
    display = vista.Display(world) # src: vista/core/Display.py

    world.reset() # Resets each of our agents, and calls the capture of each sensor that is attached for each agent.
    display.reset()

    action = follow_human_trajectory(car) # I don't think that this is really used

```

</details>
<br>

### Entities: agents and sensors

An agent (``Car.py`` in this case) of ``World`` contains one (or more) sensors. In this case, we use ``Lidar.py``. This agent is attached to ``World``, which controls everything. Most of the computation takes place after we call ``World.reset()``.

### Lidar capture process in VISTA

For our use case, we spawn a ``Lidar`` object on our Car, from ``Car.spawn_lidar()`` from ``sim_lidar.py``. When creating the ``Lidar`` sensor, we attach a ``LidarSynthesis`` object to the sensor itself.

<details>
    <summary>Here is a code snippet of Lidar.py's constructor. The synthesizer is mainly of interest for our use case.</summary>

```python
class Lidar(BaseSensor):
    """ A LiDAR sensor object that synthesizes LiDAR measurement locally around the
    dataset given a viewpoint (potentially different from the dataset) and timestamp.

    Args:
        attach_to (Entity): A car to be attached to.
        config (dict): Configuration of LiDAR sensor. An example (default) is,

            >>> DEFAULT_CONFIG = {
                'name': 'lidar_3d',
                'yaw_fov': None,
                'pitch_fov': None,
                'culling_r': 1,
                'use_synthesizer': True,
            }

            Check :class:`Lidarsynthesis` object for more details about the configuration.

    """
    DEFAULT_CONFIG = {
        'name': 'lidar_3d',
        'yaw_fov': None,
        'pitch_fov': None,
        'culling_r': 1,
        'use_synthesizer': True,
    }

    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(Lidar, self).__init__(attach_to, config)

        logging.debug('Not actually streaming lidar data when reading')
        self._streams: Dict[str, h5py.File] = dict()

        # Initialize lidar novel view synthesis object
        if self.config['use_synthesizer']:
            # Make one view synthesizer for each trace within
            # attach_to.parent.traces (different traces may have different
            # input sensors specs). For each synthesizer, pass in the input
            # and output lidar params. Then during synthesis, use the
            # appropraite synthesizer.
            self._view_synthesizers = {}
            for trace in attach_to.parent.traces:
                pfile = parse_params.ParamsFile(trace.param_file)
                in_params, _ = pfile.parse_lidar(self.name)
                self._view_synthesizers[trace] = LidarSynthesis(
                    load_model=True,
                    input_yaw_fov=in_params['yaw_fov'],
                    input_pitch_fov=in_params['pitch_fov'],
                    **self.config)

        else:
            self._view_synthesizers = None
```

</details>
<br>

It should be noted that the process of synthesizing our viewpoint here is dormant until we call the ``Lidar.capture()`` method below:

<details>
    <summary> Here is a code snippet of the Lidar.capture() method.</summary>

```python
    def capture(self, timestamp: float, **kwargs) -> np.ndarray:
        """ Synthesize LiDAR point cloud based on current timestamp and transformation
        between the novel viewpoint to be simulated and the nominal viewpoint from the
        pre-collected dataset.

        Args:
            timestamp (float): Timestamp that allows to retrieve a pointer to
                the dataset for data-driven simulation (synthesizing point cloud
                from real LiDAR sweep).

        Returns:
            np.ndarray: Synthesized point cloud.

        """
        logging.info(f'Lidar ({self.id}) capture')

        # Get frame at the closest smaller timestamp from dataset.
        multi_sensor = self.parent.trace.multi_sensor
        all_frame_nums = multi_sensor.get_frames_from_times([timestamp])

        # THIS IS WHERE WE LOAD OUR FILE FROM CONVERT_SINGLE!
        for lidar_name in multi_sensor.lidar_names:
            stream = self.streams[lidar_name]
            frame_num = all_frame_nums[lidar_name][0]
            xyz = stream['xyz'][frame_num]
            intensity = stream['intensity'][frame_num]
            pcd = Pointcloud(xyz, intensity) # src: vista/entities/sensors/lidar_utils/Pointcloud.py
            pcd = pcd[pcd.dist > 2.5]

        # Synthesis by rendering
        if self.config['use_synthesizer']:
            lat, long, yaw = self.parent.relative_state.numpy()
            logging.debug(
                f"state: {lat} {long} {yaw} \t timestamp {timestamp}")

            # We have already translated and rotated our point cloud, not used in this case.
            trans = np.array([-long, lat, 0])
            rot = np.array([0., 0, yaw]) 
            synthesizer = self._view_synthesizers[self.parent.trace]

            # THIS IS WHERE WE SYNTHESIZE OUR VIEWPOINT!
            new_pcd, new_dense = synthesizer.synthesize(trans, rot, pcd)
        else:
            new_pcd = pcd

        return new_pcd
```

</details>
<br>

When we call ``Lidar.capture()``, it eventually calls ``Lidar.LidarSynthesis.synthesize()``, which is attached to ``Car``, which is finally attached to ``World`` is where we generate our outputs.

Back to ``World``. When ``World.reset()`` is called, each agent that is defined in ``sim_lidar.py`` is reset.

<details>
  <summary>Here is a code snippet of the World.reset() method. I am pretty sure that for our use case, the loop where we reset each agent is only really useful.</summary>

```python
def reset(
    self, initial_dynamics_fn: Optional[Dict[str,
                                                Callable]] = dict()) -> None:
    """ Reset the world. This includes (1) sample a new anchor point from the real-world
    dataset to be simulated from and (2) reset states for all agents.

    Args:
        initial_dynamics_fn (Dict[str, Callable]):
            A dict mapping agent names to a function that initialize agents poses.

    """
    logging.info('World reset')

    # Sample a new trace and a new location at the sampled trace
    new_trace_index, new_segment_index, new_frame_index = \
        self.sample_new_location()

    # Reset agents
    for agent in self.agents:
        agent.reset(new_trace_index, new_segment_index, new_frame_index,
                    initial_dynamics_fn.get(agent.id, None))
```

</details>
<br>

When we call ``reset()`` to an agent, i.e., a ``Car``, we also call ``capture()`` for each sensor that is attached to the agent, after calling ``step_sensors()`` (``Car.reset()`` -> ``Car.step_sensors()`` -> call ``capture`` for each sensor attached to ``Car``). The details can be found in the source code if you want to find out things in detail.

We know that when we call ``capture()`` for each sensor that is attached on each agent, it synthesizes the final output for the sensor(s) attached.

## Files and tools

This contains information about the files/tools that are within the repository.

### Necessary files

TODO FILL LATER ON

### Tools

Contents:

1. [visualize_scene.py](#visualize_scenepy)
2. [sensorpoints.py](#sensorpointspy)
3. [trim_scenes.py (to be replaced) and remove_scenery.py](#replay_scenespy-old-to-be-replaced-and-replay_sensorpy)

#### ``visualize_scene.py``

This allows you to relate the LiDAR data from the road section into Google Maps, at a particular road point in the driver's POV.
See the code for more information of its inner workings.

**Usage**

```bash
python visualize_scene.py   # Manually prompts you to enter the trajectory
```

OR

```bash
python visualize_scene.py --trajectory {PATH_TO_TRAJECTORY}
```

You will be prompted to enter a UTM zone, and a road point into the terminal. If the output on Google Maps doesn't seem to show anything or does not represent the LiDAR data at all, you should change the UTM zone as directed.

[Back to tools](#tools)

#### ``sensorpoints,py``

This generates [points that represent the sensor FOV](README.md/#segmentation), at a particular road point, aligned with the driver's POV. It should also be noted that this code was converted from MATLAB. Note that the angular and range precisions (the spacings between each point) were forced to be 2 degrees or meters, for visibility purposes.

Note that for now, this script only supports single-sensor configurations, given in .json. Either way, both the .json and .yaml files do the same thing.

**Usage**

```bash
python sensorpoints.py  # Manually prompts you to enter the trajectory, config, and road/observer point.
```

```bash
python sensorpoints.py --trajectory {PATH_TO_TRAJECTORY} --config {PATH_TO_JSON_CONFIG} --observer_point {YOUR_OBSERVER_POINT}
```

There is also an option to regenerate your trajectory for some reason if you want to, just provide ``--regenerate True``, ``--input {PATH_TO_LAS_FILE}``, and ``--observer_height {OBSERVER_HEIGHT}`` flags in the command line.

[Back to tools](#tools)

#### ``trim_scenes.py`` (deprecated) and ``remove_scenery.py``

Performs horizontal (though fixed, along the x-axis) trimming on individual Vista scenes (trim_scenes) and horizontal trimming (follows the trajectory) on individual roads. ``trim_scenes`` should not be used since it is not flexible and only works if the vehicle is not going around a curve.

More details can be found in the code, and [here](Tools/Sceneryremoval.md).

**Usage**

To change the trimming widths (and also at specific road points), change the values in the ``generate_bounds()`` method.

```bash
python remove_scenery.py  # Manually prompts you to enter the trajectory, and road section.
```

```bash
python remove_scenery --trajectory {PATH_TO_TRAJECTORY} --input {PATH_TO_LAS_FILE}
```

[Back to tools](#tools)

#### ``replay_scenes.py`` (old, to be replaced) and ``replay_sensor.py``

Creates a slideshow, or replay of each of the Vista scenes. The point of view of the replay can be configured within the code (CTRL+F POV and the visualizer controls of Open3D can be found). **Note, Open3D must be version 0.16, and this is not currently reflected in requirements.txt (Or it might be reflected in vista_venv) for the visualizer view control to work.**

``replay_scenes.py`` only replays Vista, and currently has working video output. ``replay_sensor.py`` replays both Vista, the sensor FOV moving down the road section, and lastly (not implemented yet!) the drawing of the data rate graphs. The process to stitch all of these replays to a video has not been finished yet, and also the rendering is Open3D is really finnicky and it does need some work.

For example, the ``replay_vista()`` method does not work so far on ``replay_sensor``, only ``replay_scenes``. The ``replay_lidar`` method does work so far on ``replay_sensor`` though. Whenever you are rendering scenes using Open3D, it should follow this particular order (given that you want to render multiple scenes, ``segments`` in this case contains several ``o3d.t.geometry.PointCloud``s and make a replay):

```python
import open3d as o3d
# other imports...

# Setup our visualizer
vis = o3d.visualization.Visualizer()

vis.create_window(window_name=f"Window",
                  width=screen_width,
                  height=screen_height
                  )

vis.set_full_screen(True) # Full screen to capture full view

ctr = vis.get_view_control()

# Configure our render options here
render_opt = vis.get_render_option()
render_opt.point_size = 1.5
render_opt.background_color = np.array([16/255, 16/255, 16/255]) # 8-bit RGB, (16, 16, 16)

# Dummy geometry/pointcloud which points or colors that will be updated for each frame
current_pcd = o3d.geometry.PointCloud()

for frame, segment in enumerate(segments):
    # Update current point cloud's points and colors (for example)
    current_pcd.points = segment.to_legacy().points
    current_pcd.colors = segment.to_legacy().colors

    if frame == 0:
      vis.add_geometry(current_pcd)
      vis.reset_view_point(True)
      # Change the visualizer's field of view here using ctr.change_filed_of_view()
    else:
      vis.update_geometry(current_pcd)

    # Update the POV of the visualizer
    ctr.set_front(-traj.getForwards()[frame, :])  
    ctr.set_up(traj.getUpwards()[frame,:])
    ctr.set_lookat(traj.getRoadPoints()[frame, :] + 3.5*traj.getUpwards()[frame, :]) # Center the view around the sensor FOV
    ctr.set_zoom(0.02) # Smaller values -> more zoomed in

    # Finally update the renderer
    vis.poll_events()
    vis.update_renderer()

    # Grab your image here
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    img = (img[:,:]*255).astype(np.uint8) # Normalize RGB to 8-bit

# All done
vis.clear_geometries()
vis.clear_window()
```

**Usage**

Stuff goes here