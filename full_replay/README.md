# README

## Basic process

1. Save images from the POV of the car. For this, point clouds are read from the vista output .txt files, and then displayed on the screen using an open3d visualizer. While it is being displayed each frame is captured as an image and saved to a folder (frame_images/)
2. Save images of the top down view of the sensor. A similar process as 1., each frame is captured while the visualizer plays it on the screen. (fov/)
3. Save images of the data rate graph. Use a matplotlib animation function to play the graph on the screen, and capture the images. (plt_images/)
4. Combine all the images together, resizing as needed to create the video

## Running instructions

To run everything from scratch, use

```./full_replay.sh```

or 

```bash full_replay.sh```

This will run the whole process from scratch.

When testing changes, it is useful to save some results so you don't have to wait for a long time after each code change. So you can use

```./run.sh```

or 

```bash run.sh```

in that case. Below is an explanation of what files are saved for what parts of the process

frame_images/ : this folder contains all images of the car pov video, delete it if you want to regenerate it

fov/ : this folder contains all images of the sensor fov video, delete it if you want to regenerate it

plt_images/ : this folder contains all images of the data rate graph video, delete it if you want to regenerate it

scenes_list.pkl : this file contains the list of scenes read as point clouds, delete it if you want to regenerate it

results_cart.pkl : this file contains the list needed for generation of the simple data rate graph, delete it if you want to regenerate it.

full_replay.sh will delete everything for you and run the whole process from scratch.