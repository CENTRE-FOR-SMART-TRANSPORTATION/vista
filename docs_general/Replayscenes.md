# Scene replay documentation

[Back to README.md](README.md)

This is if you want to visualize each Vista output in a sequence, in a video format. In the future, a 'real-time' drawing of the data rate graphs, combined with a view of sensor's FOV moving down the road section are to be displayed as well.

## Overview

``replay_sensor.py`` (and also ``replay_scenes.py``, but deprecated) renders and displays each of the Vista outputs in a sequence (as well as the data rate graphs & the sensor FOV moving along the road section itself), and creates a video, stitching each element on top of the Vista outputs. I chose to use Open3D, a Python library to render and visualize each point cloud with a set POV.

### Methodology

There shouldn't be too much methodology to cover, but I will provide some information regarding the functions needed in ``replay_sensor.py``.

#### Preparing for visualization

When we want to visualize the sensor FOV as it moves through the road, we first have to read the trajectory into memory using ``file_tools.obtain_trajectory_details()`` to obtain our trajectory. We then read the .las file into memory, and then read the sensor configuration.

> The road section, or .las file is read into memory through ``open_las()``, and converted to ``o3d.geometry.PointCloud``, with a gray colormap of the .las file's intensity scalar field.

> We then use the sensor configuration, read from ``open_sensor_config_file()`` along with the ``generate_sensor_points()`` and the ``align_sensor_points()`` methods from ``sensorpoints.py`` later on during rendering to generate and align the XYZ points along each road point.

Now for the Vista scenes. The path to the Vista scenes are obtained from ``file_tools.obtain_scene_path()``.

> Vista scenes are first read into memory through ``obtain_scenes()``, in parallel. We then convert them to ``o3d.t.geometry.PointCloud`` objects so that we can visualize them later on.
>> *Note: Since we are reading our scenes into memory in parallel, the data that has been read must be pickleable. This implementation uses an opener object, because Python multiprocessing sucks.*

After reading each Vista scene into memory, we can then begin the rendering process. A temporary directory is created to store the frames of each render (sensor FOV on road, Vista scenes).

#### Rendering sensor FOV

The process to render the sensor FOV as it goes through the road is as shown (See the ``render_sensor_fov()`` method for details):

- Use Open3D to create a visualizer instance. (``vis = o3d.visualization.Visualizer()``)
- Create our window from the visualizer instance. (``vis.create_window()``)
- Obtain the view control from the visualizer instance to set the POV. (``ctr = vis.get_view_control()``)
  - NOTE: As of 5/19/2023, the ``get_view_control()`` method only returns a copy of the view control instead of a reference with version ``0.17`` of Open3D. Open3D should be installed as version ``0.16``.
- Obtain the render option from the visualizer instance to set the point size, background color, and etc. (``render_opt = vis.get_render_option()``)
- Initalize our geometries (the road itself, as well as a dummy geometry which we will update with the aligned sensor FOV) using the ``add_geometry()`` method from our visualizer instance. (``vis.add_geometry(pcd)``)
- Generate the set of sensor points representing the FOV using the ``sensorpoints.generate_sensor_fov(sensor_config)`` method. We will then align the sensor FOV points at a certain road point as we go through the for loop for rendering.
- Loop through each of the frames.
  - Align our sensor FOV points at a frame using the trajectory, and ``sensorpoints.align_sensor_points()``.
  - Change the dummy geometry's points, as well as the colors.
  - Set the visualizer POV using the view control obtained earlier and the ``set_visualizer_pov()`` method.
    - This POV can be from the driver POV, or an isometric view that can follow the driver.
  - Update the dummy geometry, then update the render to update what is displayed on the screen.
  - Finally, we then capture the rendered scene and save it to the temporary directory.

#### Rendering Vista scenes

The process to replay the Vista scenes is similar to rendering the sensor FOV as it moves through the road, except that in the visualizer instance, we are only using the dummy geometry instead of a 'background geometry'; the road.

#### Rendering the data rate graphs

This has not yet been implemented yet, but we would obtain the data rate values, then plot said values up to the current frame using matplotlib, and then save this plot to the temporary directory. There should be a plot for each rendered Vista frame saved in the temporary directory.

#### Stitching together each captured frame

This also has not yet been implemented yet, but I would stitch together each saved frame using OpenCV from the temporary directory into a video. Each frame of the video is annotated over to indicate the source files, as well as the frame number.

I took the FPS of the video as ``np.ceil((vehicle_speed/3.6)/(1*point_density))`` to make the replay itself appear as if the vehicle was going a certain speed down the road. I would probably take ``vehicle_speed`` as the speed limit in km/h for that specific road section.
