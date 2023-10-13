import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import tkinter as tk
import sys
import os
import glob
import time
import pickle

from tkinter import Tk
from pathlib import Path
from tqdm import tqdm
import utils

import matplotlib
from classes import SensorConfig, Trajectory

import file_tools

# Play our scenes using Open3D's visualizer
def visualize_replay(
    path_to_scenes: str,
    scenes_list: np.ndarray,
    vehicle_speed: np.float32 = 100,
    point_density: np.float32 = 1.0,
) -> str or int:
    # Obtain screen parameters for our video
    root = Tk()
    root.withdraw()
    SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

    # Helper function to visualize the replay of our frames in a video format
    def replay_capture_frames():
        print(f"Visualizing the scenes given by path {path_to_scenes}")

        # creates an empty point cloud on which we will display stuff using the visualizer
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        # Obtain view control of the visualizer to change POV on setup
        # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
        # only returns a copy of the view control as opposed to a reference.
        if o3d.__version__ == "0.17.0":
            print("Use the correct open3d version")
            sys.exit(1)
        else:
            ctr = vis.get_view_control()

        images_dir = os.path.join(os.getcwd(), "frame_images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        for frame, scene in enumerate(
            tqdm(scenes_list, desc="Replaying and capturing scenes")
        ):

            xyz = scene.point.positions.numpy()  # IF THE SCENE IS IN TENSOR
            geometry.points = o3d.utility.Vector3dVector(xyz)
            intensity = scene.point.colors.numpy()
            normalizer = matplotlib.colors.Normalize(
                np.min(intensity), np.max(intensity))
            las_rgb = matplotlib.cm.gray(normalizer(intensity))[:, :-1]
            geometry.colors = o3d.utility.Vector3dVector(las_rgb)

            if frame == 0:
                vis.add_geometry(geometry, reset_bounding_box=True)
            else:
                vis.update_geometry(geometry)

            # Set view of the live action Open3D replay
            if o3d.__version__ == "0.17.0":  # This probably doesn't work
                print("Use the correct open3d version")
                sys.exit(1)
            else:
                # Isometric front view
                # ctr.change_field_of_view(step=50)
                ctr.set_front([-1, 0, 0])
                ctr.set_up([0, 0, 1])
                ctr.set_lookat([18.5, 0, 1.8])
                ctr.set_zoom(0.025)

            """ Settings for POV of driver:
        ctr.set_front([-1, 0, 0])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([18.5, 0, 1.8])
        ctr.set_zoom(0.025)    
      """
            """ Settings for isometric forward POV:
        ctr.set_front([-1, -1, 1])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 1.8])
        ctr.set_zoom(0.3)  
      """

            # Update the renderer
            vis.poll_events()
            vis.update_renderer()

            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (img[:, :] * 255).astype(np.uint8)  # Normalize RGB to 8-bit

            # Capture the rendered point cloud to an RGB image for video output
            # frames.append(np.asarray(vis.capture_screen_float_buffer(do_render=True)))
            cv2.imwrite(filename=os.path.join(
                images_dir, f"{frame}.png"), img=img)

            # Play the scenes as it appears in the vehicle's speed
            # time.sleep((1*point_density)/(vehicle_speed/3.6))

        return images_dir, SCREEN_WIDTH, SCREEN_HEIGHT

    # Example taken from open3d non-blocking visualization...
    vis = o3d.visualization.Visualizer()

    while True:
        usr_inpt = input(
            f"Press 'p' to replay {len(scenes_list)} scenes given by {path_to_scenes} (press 'q' to exit), c to create video without replay (only works if you have saved images already): "
        )
        if usr_inpt == "p":
            vis.create_window(
                window_name=f"Scenes of {path_to_scenes}",
                width=SCREEN_WIDTH,
                height=SCREEN_HEIGHT,
                left=10,
                top=10,
                visible=True,
            )

            vis.set_full_screen(True)

            # View control options (also must be created befoe we can replay our frames)
            # Render options (must be c>reated before we can replay our frames)
            render_opt = vis.get_render_option()
            render_opt.point_size = 1.0
            render_opt.show_coordinate_frame = True
            render_opt.background_color = np.array(
                [16 / 255, 16 / 255, 16 / 255]
            )  # 8-bit RGB, (16, 16, 16)

            vis.poll_events()
            vis.update_renderer()

            frames, sw, sh = replay_capture_frames()
            return frames, sw, sh
        elif usr_inpt == "q":
            return
        elif usr_inpt == "c":
            images_dir = os.path.join(os.getcwd(), "frame_images")
            if not os.path.exists(images_dir):
                print("you will need to play the video to save the images")
                continue
            return images_dir, SCREEN_WIDTH, SCREEN_HEIGHT

    print("Visualization complete.")
    vis.clear_geometries()
    vis.destroy_window()

    return

def main():
    args = file_tools.parse_cmdline_args()
    path_to_scenes = file_tools.obtain_scene_path(args)
    car_path = os.path.join(os.getcwd(), "frame_images/")
    # creating the video from the pov of the driver
    # this will prompt you to click p to play, click p otherwise it won't save your video

    print("Converting the .txt scenes to point clouds...")
    scenes = utils.obtain_scenes(path_to_scenes)

    print("Creating video from car POV, please click 'p' when prompted...")
    frames, sw, sh = visualize_replay(path_to_scenes, scenes)

if __name__ == "__main__":
    main()